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

def process_instance(instance):
    """
    Processes a single instance to calculate its importance scores.
    """
    example = {}
    example["id"] = instance["id"]
    example["input"] = instance["input"]
    example["output"] = instance["output"]
    importance = [0.0]
    importance = [[0] + importance + [0]]
    
    example["importance"] = importance

    return example


if __name__ == "__main__":
    frag = Frag()
    
    with open(f"{DATA_PATH}/tasks/task3_chebi20_text2mol_selfies_test_stereo2_final.json") as f:
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
                pool.imap(process_instance, json_object["Instances"]),
                total=len(json_object["Instances"]),
            )
        )

    # 결과 병합
    for result in results:
        my_json_object["Instances"].append(result)

    print("Original data length: ", len(json_object["Instances"]))
    print("Result data length: ", len(my_json_object["Instances"]))

    with open(f"{DATA_PATH}/tasks/task3_chebi20_text2mol_selfies_test_stereo2_sim_imp.json", "w") as f:
        json.dump(my_json_object, f)
