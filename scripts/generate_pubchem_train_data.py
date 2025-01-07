import os
import sys

import selfies as sf
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm

sys.path.append(os.getcwd())
import json

from model.representation import Frag, linearize

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

if __name__ == "__main__":
    with open(
            f"{DATA_PATH}/tasks/pub_chem_data_v2.json") as f:
        json_object = json.load(f)
        
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

    fail_count = 0
    frag_set = set(["[.]"])
    frag = Frag()

    instances = json_object["Instances"]

    for instance in tqdm(instances, total=len(instances)):
        tmp_dict = {}
        tmp_dict["id"] = instance["id"]
        tmp_dict["input"] = instance["input"]
        mol = Chem.MolFromSmiles(sf.decoder(instance["output"][0][5:][:-5]))
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        raw_smiles = smiles

        linear_smiles = ""
        for smiles in raw_smiles.split("."):
            frag_str, frag_dict = linearize(smiles)
            frag_set.update(frag_dict)
            linear_smiles += frag_str + "[.]"
        linear_smiles = linear_smiles[:-3]
        tmp_dict["output"] = ["<bom>" + linear_smiles + "<eom>"]

        result_smiles = frag.decode(linear_smiles)

        if Chem.MolToInchi(
                Chem.MolFromSmiles(result_smiles)) != Chem.MolToInchi(
                    Chem.MolFromSmiles(raw_smiles)):
            print(
                f"Reconstructed SMILES {result_smiles} is not the same as the original SMILES {raw_smiles}"
            )
            fail_count += 1

        my_json_object["Instances"].append(tmp_dict)

    print(f"Fail count: {fail_count} over {len(json_object['Instances'])}")
    
    with open(
            f"{DATA_PATH}/tasks/task1_pubchem_text2mol_frag_micro_train_stereo2_camt5.json",
            "w") as f:
        json.dump(my_json_object, f)

    with open(f"asset/mol_vocabs/frag_pubchem_v2.txt", "w") as f:
        for frag in frag_set:
            f.write(frag)
            f.write("\n")
