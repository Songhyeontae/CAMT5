import os
import sys

import selfies as sf
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append("/home/osikjs/t-SMILES/t-SMILES")

from DataSet.STDTokens import CTokens, STDTokens_Frag_File
from MolUtils.RDKUtils.Frag.RDKFragUtil import Fragment_Alg
from DataSet.Graph.CNJMolAssembler import CNJMolAssembler
from DataSet.Graph.CNJMolUtil import CNJMolUtil           
from DataSet.Graph.CNJTMol import CNJMolUtils, CNJTMolTree
import json

import pandas as pd

from model.representation import T_Smiles

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

def preprocess(smiles):
    sub_smiles = smiles.strip().split('.')
    vocab_set = set()
    try:
        org_smile_sub = ''
        bfs_smile_list_sub = []
        bfs_smart_list_sub = []
        joined_sub_smiles = ''
        joined_sub_smarts = ''
        joined_sub_amt = ''
        joined_sub_idd = ''
        skeleton_sub = ''

        joined_smiles = None
        for i, sub_s in enumerate(sub_smiles):
            cnjtmol = CNJTMolTree(sub_s, ctoken = ctoken, dec_alg = Fragment_Alg.BRICS_DY) 

            if cnjtmol.mol is not None:
                for c in cnjtmol.nodes:
                    vocab_set.add(c.smiles)

                joined_amt  = cnjtmol.amt_bfs_smarts  #tsis
                joined_idd  = cnjtmol.amt_dfs_smarts

                joined_smiles, skeleton = CNJMolUtil.combine_ex_smiles(cnjtmol.bfs_ex_smiles)
                joined_smarts, _        = CNJMolUtil.combine_ex_smiles(cnjtmol.bfs_ex_smarts)

                if i > 0:
                    org_smile_sub   += '.'
                    bfs_smile_list_sub.extend(['.'])
                    bfs_smart_list_sub.extend(['.'])
                    joined_sub_smiles   += '.'
                    joined_sub_smarts   += '.'
                    skeleton_sub        += '.'
                    joined_sub_amt      += '.'
                    joined_sub_idd      += '.'

                org_smile_sub = org_smile_sub + sub_s
                bfs_smile_list_sub.extend(cnjtmol.bfs_ex_smiles)
                bfs_smart_list_sub.extend(cnjtmol.bfs_ex_smarts)

                joined_sub_smiles = joined_sub_smiles + joined_smiles
                joined_sub_smarts = joined_sub_smarts + joined_smarts
                joined_sub_amt    = joined_sub_amt + joined_amt
                joined_sub_idd    = joined_sub_idd + joined_idd

                skeleton_sub = skeleton_sub + skeleton

        if joined_smiles is None:
            return None, vocab_set

        joined_smiles = joined_smiles.strip()

    except Exception as e:
        print('[CNJTMol.preprocess].Exception:', e.args)
        print('[CNJTMol.preprocess].Exception:', smiles)
        return None, vocab_set

    return joined_sub_smiles, vocab_set

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
    vocab_set = set(["&", "^", "[n*]"])
    ctoken = CTokens(STDTokens_Frag_File(None), max_length = 256, invalid = True, onehot = False)

    for instance in tqdm(chebi20_instances, total=len(chebi20_instances)):
        tmp_dict = {}
        tmp_dict["id"] = instance["id"]
        tmp_dict["input"] = instance["input"]
        mol = Chem.MolFromSmiles(sf.decoder(instance["output"][0][5:][:-5]))
        Chem.Kekulize(mol)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        ''.join(smiles.strip().split(' '))
        tsmiles, new_vocabs = preprocess(smiles)
        vocab_set.update(new_vocabs)
        if tsmiles is None:
            tsmiles = "C"
        tmp_dict["output"] = ["<bom>" + tsmiles + "<eom>"]
        my_json_object["Instances"].append(tmp_dict)

    for row in tqdm(pubchem.itertuples(index=True), total=len(pubchem)):
        tmp_dict = {}
        tmp_dict["id"] = f"pubchem_v3_{row.Index}"
        tmp_dict["input"] = row.desc
        smiles = row.smiles
        tsmiles, new_vocabs = preprocess(smiles)
        vocab_set.update(new_vocabs)
        if tsmiles is None:
            tsmiles = "C"
        tmp_dict["output"] = ["<bom>" + tsmiles + "<eom>"]
        my_json_object["Instances"].append(tmp_dict)

    with open(
            f"{DATA_PATH}/tasks/task1_chebi20_text2mol_t_smiles_train_stereo2_w_pubchem.json",
            "w") as f:
        json.dump(my_json_object, f)

    with open(f"asset/mol_vocabs/t_smiles.txt", "w") as f:
        vocab_set = list(vocab_set)
        vocab_set.sort()
        for frag in vocab_set:
            f.write(frag)
            f.write("\n")