from model.representation import SMILES
from typing import Tuple, List, Iterator

from transformers import PreTrainedTokenizer
from model.loader import ModelLoader, Model
from model.representation import Representation

from utils import to_absolute_path
import dataclasses
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from evaluation.config import DataConfig

DESCRIPTION_TEMPLATE = (
    "Definition: You are given a molecule description in English. "
    "Your job is to generate the molecule fragments that fit the description. "
    "Now complete the following example - "
    "Input: {description} Output: "
)

class EvalDataset(Dataset):
    def __init__(self, file_path: str, chunk_size: int = 1024):
        assert file_path.endswith(".csv"), "Only CSV files are supported"
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.data_length = sum(1 for _ in open(file_path)) - 1 # Skip the header
    
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        chunk_idx = idx // self.chunk_size
        row_idx = idx % self.chunk_size
        
        chunk = pd.read_csv(self.file_path, 
                            sep="\t",
                            skiprows=chunk_idx * self.chunk_size + 1, 
                            nrows=self.chunk_size, header=None)
        
        row = chunk.iloc[row_idx]
        target, description = row[1], row[2]
        full_description = DESCRIPTION_TEMPLATE.format(description=description)

        return target, full_description

def get_dataloader(data_config: DataConfig) -> DataLoader:
    dataset = EvalDataset(
        file_path=data_config.data_path,
        chunk_size=data_config.chunk_size
    )
    return DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        prefetch_factor=data_config.prefetch_factor,
        shuffle=False
        )