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

from model.representation import Frag, linearize

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

if __name__ == "__main__":
    with open(
            f"{DATA_PATH}/tasks/task1_chebi20_text2mol_selfies_train_stereo2_final_pcdes.json"
    ) as f:
        pcdes_json_object = json.load(f)

    with open(
            f"{DATA_PATH}/tasks/task3_chebi20_text2mol_selfies_test_stereo2_final_pcdes.json"
    ) as f:
        pcdes_test_json_object = json.load(f)

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

    frag = Frag()

    # PCDes train data
    instances = pcdes_json_object["Instances"]
    for instance in tqdm(instances, total=len(instances)):
        tmp_dict = {}
        tmp_dict["id"] = instance["id"]
        tmp_dict["input"] = instance["input"]
        mol = Chem.MolFromSmiles(sf.decoder(instance["output"][0][5:][:-5]))
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)

        tmp_dict["output"] = ["<bom>" + smiles + "<eom>"]
        my_json_object["Instances"].append(tmp_dict)

    # PCDes test data for exclusion
    test_mols = set()
    instances = pcdes_test_json_object["Instances"]
    for instance in tqdm(instances, total=len(instances)):
        mol = Chem.MolFromSmiles(sf.decoder(instance["output"][0][5:][:-5]))
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        raw_smiles = smiles

        linear_smiles = ""
        for smiles in raw_smiles.split("."):
            frag_str, frag_dict = linearize(smiles)
            linear_smiles += frag_str + "[.]"
        linear_smiles = linear_smiles[:-3]
        test_mols.add(linear_smiles)

    exclude_count = 0
    # PubChem Train data
    pubchem = pd.read_csv(f"{DATA_PATH}/tasks/pub_chem_data_v3.csv", sep="\t")
    for row in tqdm(pubchem.itertuples(index=True), total=len(pubchem)):
        tmp_dict = {}
        tmp_dict["id"] = f"pubchem_v3_{row.Index}"
        tmp_dict["input"] = row.desc
        raw_smiles = row.smiles

        mol = Chem.MolFromSmiles(raw_smiles)
        linear_smiles = ""
        for smiles in raw_smiles.split("."):
            frag_str, frag_dict = linearize(smiles)
            linear_smiles += frag_str + "[.]"
        linear_smiles = linear_smiles[:-3]

        if linear_smiles in test_mols:
            print(f"Excluded {linear_smiles}")
            exclude_count += 1
            continue

        tmp_dict["output"] = ["<bom>" + raw_smiles + "<eom>"]
        my_json_object["Instances"].append(tmp_dict)

    print(f"Excluded {exclude_count} instances")
    print(
        f"Total {len(pcdes_json_object['Instances']) + len(pubchem) - exclude_count} instances"
    )

    with open(
            f"{DATA_PATH}/tasks/task1_pcdes_text2mol_smiles_train_stereo2_w_pubchem.json",
            "w") as f:
        json.dump(my_json_object, f)
