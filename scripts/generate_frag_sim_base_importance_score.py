import os
import sys

sys.path.append(os.getcwd())
import json
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

from metrics.text2mol_metrics import get_rdk_metric
from model.representation import Frag, Representation

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
        modified_tokens = tokenized_label[:i] + tokenized_label[i + 1:]

        modified = "".join(modified_tokens)
        label = "".join(tokenized_label)
        similarity = _calculate_similarity(representation, modified, label)
        importance_score = 1 - similarity
        importance_scores.append(importance_score)

    return importance_scores


def process_instance(
    instance: Dict[str, Any],
    frag: Frag = None,
):
    """
    Processes a single instance to calculate its importance scores.
    """
    example = {}
    example["id"] = instance["id"]
    example["input"] = instance["input"]
    example["output"] = instance["output"]

    encoded = example["output"][0][5:][:-5]
    importance = [0.0]
    if frag is not None:
        importance = _get_similarity_based_importance(frag, encoded)
    importance = [[0] + importance + [0]]

    example["importance"] = importance

    return example


if __name__ == "__main__":
    frag = Frag()

    print("Training set Processing...")
    with open(
            f"{DATA_PATH}/tasks/task1_chebi20_text2mol_frag_micro_train_stereo2_final.json"
    ) as f:
        json_object = json.load(f)

    train_json_object = {}
    train_json_object["Contributors"] = ["Seojin Kim"]
    train_json_object["Categories"] = ["Translation"]
    train_json_object["Reasoning"] = []
    train_json_object["URL"] = [
        "https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data"
    ]
    train_json_object["Instruction_language"] = ["English"]
    train_json_object["Domains"] = ["Chemistry_fragment"]
    train_json_object["Positive Examples"] = []
    train_json_object["Negative Examples"] = []
    train_json_object["Source"] = [
        "Translation from natural language to molecule fragments"
    ]
    train_json_object["Definition"] = [
        "You are given a molecule description in English. Your job is to generate the molecule that fits the description."
    ]
    train_json_object["Input_language"] = ["English"]
    train_json_object["Output_language"] = ["Fragment"]
    train_json_object["Instance License"] = ["Unknown"]
    train_json_object["Instances"] = []

    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(partial(process_instance, frag=frag),
                          json_object["Instances"]),
                total=len(json_object["Instances"]),
            ))

    for result in results:
        train_json_object["Instances"].append(result)

    with open(
            f"{DATA_PATH}/tasks/task1_chebi20_text2mol_frag_micro_train_stereo2_sim_imp.json",
            "w") as f:
        json.dump(train_json_object, f)

    print("Validation set Processing...")
    with open(
            f"{DATA_PATH}/tasks/task2_chebi20_text2mol_selfies_validation_stereo2_final.json"
    ) as f:
        json_object = json.load(f)

    validation_json_object = {}
    validation_json_object["Contributors"] = ["Seojin Kim"]
    validation_json_object["Categories"] = ["Translation"]
    validation_json_object["Reasoning"] = []
    validation_json_object["URL"] = [
        "https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data"
    ]
    validation_json_object["Instruction_language"] = ["English"]
    validation_json_object["Domains"] = ["Chemistry_fragment"]
    validation_json_object["Positive Examples"] = []
    validation_json_object["Negative Examples"] = []
    validation_json_object["Source"] = [
        "Translation from natural language to molecule fragments"
    ]
    validation_json_object["Definition"] = [
        "You are given a molecule description in English. Your job is to generate the molecule that fits the description."
    ]
    validation_json_object["Input_language"] = ["English"]
    validation_json_object["Output_language"] = ["Fragment"]
    validation_json_object["Instance License"] = ["Unknown"]
    validation_json_object["Instances"] = []

    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(process_instance, json_object["Instances"]),
                total=len(json_object["Instances"]),
            ))

    for result in results:
        validation_json_object["Instances"].append(result)

    with open(
            f"{DATA_PATH}/tasks/task2_chebi20_text2mol_selfies_validation_stereo2_sim_imp.json",
            "w") as f:
        json.dump(validation_json_object, f)

    print("Test set Processing...")
    with open(
            f"{DATA_PATH}/tasks/task3_chebi20_text2mol_selfies_test_stereo2_final.json"
    ) as f:
        json_object = json.load(f)

    test_json_object = {}
    test_json_object["Contributors"] = ["Seojin Kim"]
    test_json_object["Categories"] = ["Translation"]
    test_json_object["Reasoning"] = []
    test_json_object["URL"] = [
        "https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data"
    ]
    test_json_object["Instruction_language"] = ["English"]
    test_json_object["Domains"] = ["Chemistry_fragment"]
    test_json_object["Positive Examples"] = []
    test_json_object["Negative Examples"] = []
    test_json_object["Source"] = [
        "Translation from natural language to molecule fragments"
    ]
    test_json_object["Definition"] = [
        "You are given a molecule description in English. Your job is to generate the molecule that fits the description."
    ]
    test_json_object["Input_language"] = ["English"]
    test_json_object["Output_language"] = ["Fragment"]
    test_json_object["Instance License"] = ["Unknown"]
    test_json_object["Instances"] = []

    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(process_instance, json_object["Instances"]),
                total=len(json_object["Instances"]),
            ))

    for result in results:
        test_json_object["Instances"].append(result)

    with open(
            f"{DATA_PATH}/tasks/task3_chebi20_text2mol_selfies_test_stereo2_sim_imp.json",
            "w") as f:
        json.dump(test_json_object, f)
