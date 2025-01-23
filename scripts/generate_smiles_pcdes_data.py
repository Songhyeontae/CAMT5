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
    # Train data
    with open(
            f"{DATA_PATH}/tasks/task1_chebi20_text2mol_selfies_train_stereo2_final_pcdes.json"
    ) as f:
        json_object = json.load(f)

    train_json_obejct = {}
    train_json_obejct["Contributors"] = ["Seojin Kim"]
    train_json_obejct["Categories"] = ["Translation"]
    train_json_obejct["Reasoning"] = []
    train_json_obejct["URL"] = [
        "https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data"
    ]
    train_json_obejct["Instruction_language"] = ["English"]
    train_json_obejct["Domains"] = ["Chemistry_fragment"]
    train_json_obejct["Positive Examples"] = []
    train_json_obejct["Negative Examples"] = []
    train_json_obejct["Source"] = [
        "Translation from natural language to molecule fragments"
    ]
    train_json_obejct["Definition"] = [
        "You are given a molecule description in English. Your job is to generate the molecule that fits the description."
    ]
    train_json_obejct["Input_language"] = ["English"]
    train_json_obejct["Output_language"] = ["Fragment"]
    train_json_obejct["Instance License"] = ["Unknown"]

    train_json_obejct["Instances"] = []

    instances = json_object["Instances"]

    for instance in tqdm(instances, total=len(instances)):
        tmp_dict = {}
        tmp_dict["id"] = instance["id"]
        tmp_dict["input"] = instance["input"]
        mol = Chem.MolFromSmiles(sf.decoder(instance["output"][0][5:][:-5]))
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)

        tmp_dict["output"] = [smiles]
        train_json_obejct["Instances"].append(tmp_dict)

    print(f"{len(json_object['Instances'])} objects are processed.")

    with open(
            f"{DATA_PATH}/tasks/task1_pcdes_text2mol_smiles_train_stereo2.json",
            "w") as f:
        json.dump(train_json_obejct, f)
        
    # Validation data
    with open(
            f"{DATA_PATH}/tasks/task2_chebi20_text2mol_selfies_validation_stereo2_final_pcdes.json"
    ) as f:
        json_object = json.load(f)

    val_json_obejct = {}
    val_json_obejct["Contributors"] = ["Seojin Kim"]
    val_json_obejct["Categories"] = ["Translation"]
    val_json_obejct["Reasoning"] = []
    val_json_obejct["URL"] = [
        "https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data"
    ]
    val_json_obejct["Instruction_language"] = ["English"]
    val_json_obejct["Domains"] = ["Chemistry_fragment"]
    val_json_obejct["Positive Examples"] = []
    val_json_obejct["Negative Examples"] = []
    val_json_obejct["Source"] = [
        "Translation from natural language to molecule fragments"
    ]
    val_json_obejct["Definition"] = [
        "You are given a molecule description in English. Your job is to generate the molecule that fits the description."
    ]
    val_json_obejct["Input_language"] = ["English"]
    val_json_obejct["Output_language"] = ["Fragment"]
    val_json_obejct["Instance License"] = ["Unknown"]

    val_json_obejct["Instances"] = []

    instances = json_object["Instances"]

    for instance in tqdm(instances, total=len(instances)):
        tmp_dict = {}
        tmp_dict["id"] = instance["id"]
        tmp_dict["input"] = instance["input"]
        mol = Chem.MolFromSmiles(sf.decoder(instance["output"][0][5:][:-5]))
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)

        tmp_dict["output"] = [smiles]
        val_json_obejct["Instances"].append(tmp_dict)

    print(f"{len(json_object['Instances'])} objects are processed.")

    with open(
            f"{DATA_PATH}/tasks/task2_pcdes_text2mol_smiles_validation_stereo2.json",
            "w") as f:
        json.dump(val_json_obejct, f)
        
    # Test data
    with open(
            f"{DATA_PATH}/tasks/task3_chebi20_text2mol_selfies_test_stereo2_final_pcdes.json"
    ) as f:
        json_object = json.load(f)

    test_json_obejct = {}
    test_json_obejct["Contributors"] = ["Seojin Kim"]
    test_json_obejct["Categories"] = ["Translation"]
    test_json_obejct["Reasoning"] = []
    test_json_obejct["URL"] = [
        "https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data"
    ]
    test_json_obejct["Instruction_language"] = ["English"]
    test_json_obejct["Domains"] = ["Chemistry_fragment"]
    test_json_obejct["Positive Examples"] = []
    test_json_obejct["Negative Examples"] = []
    test_json_obejct["Source"] = [
        "Translation from natural language to molecule fragments"
    ]
    test_json_obejct["Definition"] = [
        "You are given a molecule description in English. Your job is to generate the molecule that fits the description."
    ]
    test_json_obejct["Input_language"] = ["English"]
    test_json_obejct["Output_language"] = ["Fragment"]
    test_json_obejct["Instance License"] = ["Unknown"]

    test_json_obejct["Instances"] = []

    instances = json_object["Instances"]

    for instance in tqdm(instances, total=len(instances)):
        tmp_dict = {}
        tmp_dict["id"] = instance["id"]
        tmp_dict["input"] = instance["input"]
        mol = Chem.MolFromSmiles(sf.decoder(instance["output"][0][5:][:-5]))
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)

        tmp_dict["output"] = [smiles]
        test_json_obejct["Instances"].append(tmp_dict)

    print(f"{len(json_object['Instances'])} objects are processed.")

    with open(
            f"{DATA_PATH}/tasks/task3_pcdes_text2mol_smiles_test_stereo2.json",
            "w") as f:
        json.dump(test_json_obejct, f)
