import os
import sys
sys.path.append(os.getcwd())
from multiprocessing import Pool, cpu_count
from functools import partial

from model.representation import Frag, Representation
from typing import List
import json
from tqdm import tqdm
from metrics.text2mol_metrics import get_rdk_metric
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
DUMMY_FRAG = "[C]"

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

def _get_similarity_based_importance(
    representation: Representation,
    label_text: str,
) -> List[float]:

    importance_scores = []  # Store importance scores for the current label

    
    tokenized_label = _split_outer_brackets(label_text)

    for i in range(len(tokenized_label)):
        modified_tokens = tokenized_label[:i] + tokenized_label[i+1:]
        
        modified = "".join(modified_tokens)
        label = "".join(tokenized_label)
        similarity = _calculate_similarity(representation, modified, label)
        importance_score = 1 - similarity
        importance_scores.append(importance_score)

    return importance_scores

def process_instance(instance, frag):
    """
    Processes a single instance to calculate its importance scores.
    """
    example = {}
    example["id"] = instance["id"]
    example["input"] = instance["input"]
    example["output"] = instance["output"]

    encoded = example["output"][0][5:][:-5]
    importance = _get_similarity_based_importance(frag, encoded)
    importance = [[0] + importance + [0]]
    
    example["importance"] = importance

    return example


if __name__ == "__main__":
    frag = Frag()
    
    with open(f"{DATA_PATH}/tasks/task1_chebi20_text2mol_frag_micro_train_stereo2_final.json") as f:
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

    # 병렬 처리 함수 정의
    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(partial(process_instance, frag=frag), json_object["Instances"]),
                total=len(json_object["Instances"]),
            )
        )

    # 결과 병합
    for result in results:
        my_json_object["Instances"].append(result)

    print("Original data length: ", len(json_object["Instances"]))
    print("Result data length: ", len(my_json_object["Instances"]))

    with open(f"{DATA_PATH}/tasks/task1_chebi20_text2mol_frag_micro_train_stereo2_sim_imp.json", "w") as f:
        json.dump(my_json_object, f)
