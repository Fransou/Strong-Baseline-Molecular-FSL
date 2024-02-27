"""File taking each dataset in data/LIT-PCBA and converting it to a csv file with a column Drug containing the smiles of the dataset, and a column Y containing the labels of the dataset."""

import os
import pandas as pd
import numpy as np
import argparse
import logging
import time
import warnings
import sys

from tqdm import tqdm

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "few_shot_drug", "TDC_tasks", "data")
LIT_PCBA_PATH = os.path.join(DATA_PATH, "LIT-PCBA")
LIT_PCBA_CSV_PATH = os.path.join(DATA_PATH, "HTS")


print("Running lit_pcba_to_csv.py")

for dir in tqdm(os.listdir(LIT_PCBA_PATH)):
    #open actives.smi and inactives.smi
    actives_path = os.path.join(LIT_PCBA_PATH, dir, "actives.smi")
    inactives_path = os.path.join(LIT_PCBA_PATH, dir, "inactives.smi")
    actives = pd.read_csv(actives_path, sep=" ", header=None, names=["Drug", "Drug_ID"])
    actives["Y"] = 1
    inactives = pd.read_csv(inactives_path, sep=" ", header=None, names=["Drug", "Drug_ID"])
    inactives["Y"] = 0

    #add column Y with 1 for actives and 0 for inactives
    df = pd.concat([actives, inactives])
    df.to_csv(os.path.join(LIT_PCBA_CSV_PATH, f"{dir}.csv"), index=False)