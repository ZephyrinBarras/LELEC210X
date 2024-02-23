import pickle
from pathlib import Path
from typing import Optional

import click

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
from numpy import *
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from .utils import payload_to_melvecs

load_dotenv()


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
    pca = pickle.load(open("classification/data/models/pca_Q1_parameters","rb"))
    model = pickle.load(open("classification/data/models/random_forest_Q1_parameters.pickle","rb"))

    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX) :]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
            #logger.info(f"Parsed payload into Mel vectors: {melvecs}")
            "logger.info(str(len(melvecs))+str( len(melvecs[0])))
            #logger.info("ananas")
            melvecs = melvecs.reshape(-1)
            "logger.info(f"Parsed payload into Mel vectors: {melvecs}")
            mp = pca.transform([melvecs[:-20]])
            #logger.info(f"Parsed payload into Mel vectors: {mp}")
            classe = model.predict(mp)
            logger.info(f"Parsed payload into Mel vectors: {classe}")

            
