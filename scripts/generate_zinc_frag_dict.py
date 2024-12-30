import os
import sys

import selfies as sf
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm

sys.path.append(os.getcwd())
from concurrent.futures import ProcessPoolExecutor, as_completed

import datasets

from model.representation import Frag, linearize

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")


def process_example(example):
    try:
        mol = Chem.MolFromSmiles(example["smiles"])
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        raw_smiles = smiles

        frag_set_local = set()

        for smiles in raw_smiles.split("."):
            _, frag_dict = linearize(smiles)
            frag_set_local.update(frag_dict)

        return frag_set_local, 1
    except:
        return set(), 0


def batchify(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


if __name__ == "__main__":
    zinc_dataset = datasets.load_dataset("zpn/zinc20",
                                         split="train",
                                         streaming=True)
    dataset_info = datasets.get_dataset_infos("zpn/zinc20")
    print("Dataset info:", dataset_info)
    train_size = dataset_info["default"].splits["train"].num_examples
    print("Train size:", train_size)

    frag_set = set(["[.]"])
    frag = Frag()
    fail_count = 0

    BATCH_SIZE = 768
    futures = []

    with ProcessPoolExecutor() as executor:
        progress_bar = tqdm(total=train_size)
        for batch in batchify(zinc_dataset, BATCH_SIZE):
            for example in batch:
                futures.append(executor.submit(process_example, example))

            for future in as_completed(futures):
                try:
                    frag_sub_set, fail = future.result()
                    frag_set.update(frag_sub_set)
                    fail_count += fail
                    progress_bar.update(1)
                except Exception as e:
                    print(f"Error: {e}")
                    fail_count += 1
                finally:
                    futures.remove(future)

        progress_bar.close()

    print(f"Fail count: {fail_count}")

    # for example in tqdm(zinc_dataset):
    #     mol = Chem.MolFromSmiles(example["smiles"])
    #     smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
    #     raw_smiles = smiles

    #     try:
    #         for smiles in raw_smiles.split("."):
    #             _, frag_dict = linearize(smiles)
    #             frag_set.update(frag_dict)
    #     except:
    #         fail_count += 1
    #         continue

    print(f"Fail count: {fail_count}")

    with open(f"asset/mol_vocabs/frag_zinc20_camt5.txt", "w") as f:
        for frag in frag_set:
            f.write(frag)
            f.write("\n")
