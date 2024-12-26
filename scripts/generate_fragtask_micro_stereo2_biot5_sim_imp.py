import os
import sys
sys.path.append(os.getcwd())

from model.representation import Frag, Representation
from typing import List
import json
from tqdm import tqdm
import selfies as sf
from rdkit import Chem
from metrics.text2mol_metrics import get_rdk_metric
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
DUMMY_FRAG = "[C]"

from model.loader import ModelLoader
from model.config import ModelConfig, LoadModel, TokenizerConfig

def _calculate_similarity(
    representation: Representation,
    modified_txt: str,
    label_txt: str,
) -> float:
    """
    Calculates similarity between two texts using cosine similarity.
    """
    modified = representation.decode(modified_txt)
    label = representation.decode(label_txt)
    similarity = get_rdk_metric(modified, label)
    return similarity

def _split_outer_brackets(s):
    result = []
    current = []
    depth = 0

    for char in s:
        if char == '[':
            if depth == 0:
                # Start of a new outer bracket
                current = []
            depth += 1
        if depth > 0:
            # Append characters inside the current bracket
            current.append(char)
        if char == ']':
            depth -= 1
            if depth == 0:
                # End of the current outer bracket
                result.append(''.join(current))
                current = []
    return result

def _tokenize(s):
    model_config = ModelConfig(
        name="google/t5-v1_1-base",
        load_model=LoadModel(from_pretrained=False),
        tokenizer_config=TokenizerConfig(
            additional_tokens_paths=["asset/mol_vocabs/frag_stereo.txt", "asset/mol_vocabs/selfies.txt"]
        ),
        representation_type="frag",
    )
    loader = ModelLoader(model_config)
    tokenizer = loader.load_tokenizer()
    
    tokenized = tokenizer.tokenize(s)
    
    return tokenized
    
def _get_similarity_based_importance(
    representation: Representation,
    label_text: str,
    raw_smiles: str,
    id,
) -> List[float]:

    importance_scores = []  # Store importance scores for the current label

    tokenized_label = _split_outer_brackets(label_text)
    tokenized_label_2 = _tokenize(label_text)
    
    if tokenized_label != tokenized_label_2:
        print(f"label_text: {label_text}")
        print(f"Raw smiles: {raw_smiles}")
        print(f"ID: {id}")
        print(f"Tokenized labels are different: {tokenized_label} vs {tokenized_label_2}")
        return None

    # for i in range(len(tokenized_label)):
    #     modified_tokens = tokenized_label[:i] + tokenized_label[i+1:]
        
    #     modified = "".join(modified_tokens)
    #     label = "".join(tokenized_label)
    #     similarity = _calculate_similarity(representation, modified, label)
    #     importance_score = 1 - similarity
    #     importance_scores.append(importance_score)

    importance_scores = [0] * len(tokenized_label)
    return importance_scores

with open(f"{DATA_PATH}/tasks/task1_chebi20_text2mol_selfies_train_stereo2_final.json") as f:
    json_object = json.load(f)

my_json_object = {}
my_json_object["Contributors"] = ["Seojin Kim"]
my_json_object["Categories"] = ["Translation"]
my_json_object["Reasoning"] = []
my_json_object["URL"] = ["https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data"]
my_json_object["Instruction_language"] = ["English"]
my_json_object["Domains"] = ["Chemistry_fragment"]
my_json_object["Positive Examples"] = []
my_json_object["Negative Examples"] = []
my_json_object["Source"] = ["Translation from natural language to molecule fragments"]
my_json_object["Definition"] = ["You are given a molecule description in English. Your job is to generate the molecule that fits the description."]
my_json_object["Input_language"] = ["English"]
my_json_object["Output_language"] = ["Fragment"]
my_json_object["Instance License"] = ["Unknown"]
my_json_object["Instances"] = []
my_json_object["Importances"] = []

incorrect = 0
correct = 0
invalid = 0
wrong_cnt = 0

for instance in tqdm(json_object["Instances"], total=len(json_object["Instances"])):
    example = {}
    example["id"] = instance["id"]
    example["input"] = instance["input"]
    mol = Chem.MolFromSmiles(sf.decoder(instance["output"][0][5:][:-5]))
    smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
    raw_smiles = smiles
    
    frag = Frag()
    encoded = frag.encode(smiles)
    if encoded == DUMMY_FRAG:
        invalid += 1
        incorrect += 1
        continue
    
    decoded_smiles = frag.decode(encoded)
    if Chem.MolToInchi(Chem.MolFromSmiles(raw_smiles)) == Chem.MolToInchi(Chem.MolFromSmiles(decoded_smiles)):
        correct += 1
        
    else:
        incorrect += 1
        continue
    
    example["output"] = ["<bom>" + encoded + "<eom>"]
    importance = _get_similarity_based_importance(frag, encoded, raw_smiles, instance["id"])
    
    if importance is None:
        wrong_cnt += 1
    # importance = [0] + importance + [0]
    
    # print(f"Importance length: {len(importance)}")
    my_json_object["Instances"].append(example)
    # my_json_object["Importances"].append(importance)
    
            
print("correct", correct)
print("incorrect", incorrect)
print("invalid", invalid)
print(f"Wrong count: {wrong_cnt}")

print("Original data length: ", len(json_object["Instances"]))
print("Result data length: ", len(my_json_object["Instances"]))
# with open(f"{DATA_PATH}/tasks/task1_chebi20_text2mol_frag_micro_train_stereo2_sim_imp.json", "w") as f:
#     json.dump(my_json_object, f)



