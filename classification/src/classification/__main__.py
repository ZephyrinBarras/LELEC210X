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
    melvec_length=30 #harcode simplicité
    nbr_packet = 0
    lim  =0
    data = []

    for payload in _input:
        nbr_packet+=1
        
        
        if PRINT_PREFIX in payload:
            previous = 0

            #RECUPERE ET TRAITE LE MEL
            #serial_number = int(payload[len(PRINT_PREFIX):len(PRINT_PREFIX)+5],16)
            
            #packet_number = int(payload[len(PRINT_PREFIX):len(PRINT_PREFIX)+7],16)%20
            payload = payload[len(PRINT_PREFIX)+7:]
            melvecs = payload_to_melvecs(payload, melvec_length, 1)
            print("noise level",melvecs[-1]/3.05175781e-05)
            print("melvec",melvecs[:-1]/3.05175781e-05)
            melvecs = reshape(melvecs[:-1] , (1,29))
            melvecs = melvecs/linalg.norm(melvecs)
            if lim<50:
                data.append(melvecs[0])
                lim+=1
            else:
                plt.imshow(data)
                plt.show()
            model = pickle.load(open("classification/data/models/model_pca_29.pickle","rb"))
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            elem = model.predict_proba(melvecs)
            
            print(elem)
            ajouter(elem)
            global commencement
            commencement+=1
            probs=memory
            """maxvrai= zeros(5) 
            for m in range(5) :
                for n in range(len(probs)):
                    maxvrai[m]=+probs[n,m]"""
            #pour le seuille
            """seuille=0.6
            if max(10**(maxvrai/5))<seuille:
                classe=" "
            else:
                classe=name[argmax(maxvrai)]"""
            classe=name[argmax(elem)]
        
            logger.info(f"Parsed payload into Mel vectors: {classe}")
            #main()