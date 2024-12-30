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
            f"{DATA_PATH}/tasks/pub_chem_data.json") as f:
        json_object = json.load(f)

    fail_count = 0
    frag_set = set(["[.]"])
    frag = Frag()

    for instance in tqdm(json_object["Instances"],
                         total=len(json_object["Instances"])):
        mol = Chem.MolFromSmiles(sf.decoder(instance["output"][0][5:][:-5]))
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        raw_smiles = smiles

        linear_smiles = ""
        for smiles in raw_smiles.split("."):
            frag_str, frag_dict = linearize(smiles)
            frag_set.update(frag_dict)
            linear_smiles += frag_str + "[.]"
        linear_smiles = linear_smiles[:-3]

        result_smiles = frag.decode(linear_smiles)

        if Chem.MolToInchi(
                Chem.MolFromSmiles(result_smiles)) != Chem.MolToInchi(
                    Chem.MolFromSmiles(raw_smiles)):
            print(
                f"Reconstructed SMILES {result_smiles} is not the same as the original SMILES {raw_smiles}"
            )
            fail_count += 1

    print(f"Fail count: {fail_count} over {len(json_object['Instances'])}")

    with open(f"asset/mol_vocabs/frag_pubchem_camt5.txt", "w") as f:
        for frag in frag_set:
            f.write(frag)
            f.write("\n")
