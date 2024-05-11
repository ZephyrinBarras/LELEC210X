from pathlib import Path
from typing import Optional

import click

from auth import PRINT_PREFIX
import pickle as pickle
from common.logging import logger
from numpy import *
import numpy as np

from pathlib import Path
from typing import Optional


import click

from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
from numpy import *
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from .utils import payload_to_melvecs
import requests

def send(classe):
    hostname = "http://lelec210x.sipr.ucl.ac.be/lelec210x/"
    key = "QMJf02nSY3SvesXt1I8Vi9bKZT520NmM4LpIhHNN"
    response = requests.post(f"{hostname}/leaderboard/submit/{key}/{classe}", timeout=1)


listeclass=["birds","chainsaw","fire","handsaw","helicopter"]
name = sorted(listeclass)
max_size = 4 #à changer
memory = np.zeros((max_size,5))
pos = 0
commencement=0
count = 0
max_count = 300
data = []
vote = np.zeros(5)
#model = pickle.load(open("./model_def.pickle","rb"))
parity= 0
memory_size = 5
memory_value = np.zeros((memory_size,len(name)))
position = 0
vote = np.zeros(5)
classe_vote = []
def ajouter(elem):
    global pos
    memory[pos] = elem
    pos = (pos+1)%max_size
    
def get_prob(prob):
    result =np.mean(memory)
    print(result)
    return np.mean(memory)

def result():
    return np.argmax(get_prob())


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
def main(
    _input: Optional[click.File],
    model: Optional[Path],
) -> None:
    """
    Extract Mel vectors from payloads and perform classification on them.
    Classify MELVECs contained in payloads (from packets).
    """
    model = pickle.load(open("classification/data/models/random_forest_Q1_parameters.pickle","rb"))
    count = 0
    max_count = 300
    data = []
    #model = pickle.load(open("./model_def.pickle","rb"))
    parity= 0
    memory_size = 5
    memory_value = np.zeros((memory_size,len(name)))
    position = 0
    
    classe_vote = []

    def voteur(array):
        global vote, name
        for k in range(len(vote)):
            count = 0
            for i in array:
                if i==name[k]:
                    count+=1
            vote[k] = count

    model = pickle.load(open("./model_def.pickle","rb"))
    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload_to_melvecs(payload[len(PRINT_PREFIX):],24,1)

            # Print the contents of the serial data
            temp =payload
            feature=np.zeros((2,24))    #24=MEL_LENGTH
            feature[parity] = temp[0]
            parity = (parity+1)%2
            if parity:
                elem = model.predict_proba([np.ravel(feature)])
                predict = model.predict([np.ravel(feature)])
                print(elem, model.predict([np.ravel(feature)]))
                classe_vote.append(predict[0])
                if len(classe_vote)>10:
                    del(classe_vote[0])
                voteur(classe_vote)


                if np.max(vote)>=4:
                    print("critere vote")
                    print("Envoie de : ", name[np.argmax(vote)])
                    send(name[np.argmax(vote)])

                if np.max(elem)>0.8:
                    print("je déconne à fond")
                    send(name[np.argmax(vote)])
                loga = np.log(elem)
                memory_value[position] = loga

                position=(position+1)%5
                proba_sum = np.sum(memory_value, axis=0)
                retour = np.exp(proba_sum/5)
                var = np.var(retour)

                if np.max(np.exp(proba_sum/5))>0.65:
                    print("nouveau threshold")
                    send(name[np.argmax(vote)])

                trie = np.flip(np.sort(retour))
                if np.var(var>0.2 and trie[0]-trie[1]>0.2):
                    print("treash variance")
                    send(name[np.argmax(vote)])
                if np.var(trie[0]-trie[1]>0.3):
                    print("treash diff max")
                    send(name[np.argmax(vote)])
            
