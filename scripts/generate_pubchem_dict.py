import os
import sys

from dotenv import load_dotenv
from rdkit import Chem
from tqdm import tqdm

sys.path.append(os.getcwd())
import pandas as pd

from model.representation import Frag, linearize

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

if __name__ == "__main__":
    df = pd.read_csv(f"{DATA_PATH}/tasks/pub_chem_data_v3.csv",
                     sep="\t",
                     usecols=["smiles"])
    fail_count = 0
    frag_set = set(["[.]"])
    frag = Frag()

    for smiles in tqdm(df["smiles"], total=len(df)):
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

    print(f"Fail count: {fail_count} over {len(df)}")

    with open(f"asset/mol_vocabs/frag_pubchem_v3.txt", "w") as f:
        for frag in frag_set:
            f.write(frag)
            f.write("\n")
