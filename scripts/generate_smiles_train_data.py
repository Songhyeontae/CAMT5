import os
import sys

import selfies as sf
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm

sys.path.append(os.getcwd())
import json
import pandas as pd

from model.representation import Smiles

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

if __name__ == "__main__":
    with open(
            f"{DATA_PATH}/tasks/task1_chebi20_text2mol_selfies_train_stereo2_final.json"
    ) as f:
        chebi20_json_object = json.load(f)
    pubchem = pd.read_csv(f"{DATA_PATH}/tasks/pub_chem_data_v3.csv", sep="\t")

    my_json_object = {}
    my_json_object["Contributors"] = ["Seojin Kim"]
    my_json_object["Categories"] = ["Translation"]
    my_json_object["Reasoning"] = []
    my_json_object["URL"] = [
        "https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data"
    ]
    my_json_object["Instruction_language"] = ["English"]
    my_json_object["Domains"] = ["Chemistry_fragment"]
    my_json_object["Positive Examples"] = []
    my_json_object["Negative Examples"] = []
    my_json_object["Source"] = [
        "Translation from natural language to molecule fragments"
    ]
    my_json_object["Definition"] = [
        "You are given a molecule description in English. Your job is to generate the molecule that fits the description."
    ]
    my_json_object["Input_language"] = ["English"]
    my_json_object["Output_language"] = ["Fragment"]
    my_json_object["Instance License"] = ["Unknown"]

    my_json_object["Instances"] = []
    chebi20_instances = chebi20_json_object["Instances"]

    for instance in tqdm(chebi20_instances, total=len(chebi20_instances)):
        tmp_dict = {}
        tmp_dict["id"] = instance["id"]
        tmp_dict["input"] = instance["input"]
        mol = Chem.MolFromSmiles(sf.decoder(instance["output"][0][5:][:-5]))
        Chem.Kekulize(mol)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)

        tmp_dict["output"] = ["<bom>" + smiles + "<eom>"]
        my_json_object["Instances"].append(tmp_dict)
        
    for row in tqdm(pubchem.itertuples(index=True), total=len(pubchem)):
        tmp_dict = {}
        tmp_dict["id"] = f"pubchem_v3_{row.Index}"
        tmp_dict["input"] = row.desc
        smiles = row.smiles
        tmp_dict["output"] = ["<bom>" + smiles + "<eom>"]
        my_json_object["Instances"].append(tmp_dict)

    with open(
            f"{DATA_PATH}/tasks/task1_chebi20_text2mol_smiles_train_stereo2_w_pubchem.json",
            "w") as f:
        json.dump(my_json_object, f)
