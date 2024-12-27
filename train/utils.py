import math
import string
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import torch
from transformers.data.data_collator import *

from model.representation import Representation
from train.config import TokenImportance, TokenImportanceConfig
from utils import to_absolute_path

IMPORTANCE_PAD_VALUE = -1.0


@dataclass
class DataCollatorForText2Mol:
    tokenizer: PreTrainedTokenizerBase
    representation: Representation
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    token_importance_config: TokenImportanceConfig = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "

            definition = ""
            if isinstance(instance["Definition"], list):
                definition = "Definition: " + instance["Definition"][0].strip()
            else:
                definition = "Definition: " + instance["Definition"].strip()
            if not definition[-1] in string.punctuation:
                definition += "."
            definition += "\n\n"

            source = definition + task_input
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(
                    self.tokenizer.decode(
                        tokenized_source[:self.max_source_length],
                        skip_special_tokens=True,
                    ))

        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        assert all(ex["Instance"].get("output") != None for ex in batch), \
            "output is required for data in TextToMol task."

        labels = [ex["Instance"]["output"][0] for ex in batch]
        token_importances = None
        if "importance" in batch[0]["Instance"]:
            token_importances = [
                ex["Instance"]["importance"][0] for ex in batch
            ]

        encoded_labels = self.tokenizer(
            labels,
            max_length=self.max_target_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        label_mask = encoded_labels["attention_mask"].bool()
        model_inputs["labels"] = encoded_labels["input_ids"].masked_fill(
            ~label_mask, self.label_pad_token_id)
        max_seq_length = model_inputs["labels"].shape[1]

        if self.token_importance_config is not None:
            token_importanaces = get_token_importance(
                config=self.token_importance_config,
                labels=labels,
                tokenizer=self.tokenizer,
                representation=self.representation,
                predefined_importances=token_importances,
            )
            model_inputs["token_importances"] = _pad_list_to_tensor(
                token_importanaces, max_seq_length, IMPORTANCE_PAD_VALUE)

        return model_inputs


def get_token_importance(
    config: TokenImportanceConfig,
    labels: List[str],
    tokenizer: PreTrainedTokenizerBase,
    representation: Representation,
    predefined_importances: Optional[List[List[float]]] = None,
) -> List[List[float]]:

    tokenized_labels = [
        tokenizer.tokenize(label) + [tokenizer.eos_token] for label in labels
    ]

    if config.token_importance == TokenImportance.ATOM_COUNT.value:
        token_importances = _get_atom_count_importances(
            tokenized_labels,
            tokenizer,
            representation,
            config.special_token_importance,
        )
    elif config.token_importance == TokenImportance.ATOM_FREQ.value:
        token_importances = _get_atom_freq_importances(
            tokenized_labels,
            tokenizer,
            representation,
            config.atom_freq_path,
            config.special_token_importance,
        )
    elif config.token_importance == TokenImportance.PREDEFINED.value:
        token_importances = _get_predefined_importances(
            tokenizer,
            tokenized_labels,
            predefined_importances,
            config.special_token_importance,
        )
    else:
        raise ValueError(
            f"Invalid token importance type: {config.token_importance}")

    return token_importances


def _get_atom_count_importances(
    tokenized_labels: List[List[str]],
    tokenizer: PreTrainedTokenizerBase,
    representation: Representation,
    special_token_importance: float,
) -> List[List[float]]:

    token_importances = []
    for label in tokenized_labels:
        token_importance = []
        for token in label:
            if token in tokenizer.all_special_tokens:
                token_importance.append(special_token_importance)
            else:
                token_importance.append(representation.get_size(token))
        token_importances.append(token_importance)
    return token_importances


def _get_atom_freq_importances(
    tokenized_labels: List[List[str]],
    tokenizer: PreTrainedTokenizerBase,
    representation: Representation,
    atom_freq_path: str,
    special_token_importance: float,
) -> List[List[float]]:
    atom_freq_scores = _get_atom_freq_score(atom_freq_path)

    token_importances = []
    for label in tokenized_labels:
        token_importance = []
        for token in label:
            if token in tokenizer.all_special_tokens:
                token_importance.append(special_token_importance)
            else:
                token_importance.append(
                    representation.get_atom_weighted_score(
                        token, atom_freq_scores))
        token_importances.append(token_importance)
    return token_importances


@lru_cache(maxsize=1)
def _get_atom_freq_score(atom_freq_path: str) -> Dict[str, float]:
    if atom_freq_path is None:
        raise ValueError("Please provide atom frequency path")

    atom_freq_path = to_absolute_path(atom_freq_path)
    atom_freqs = {}
    with open(atom_freq_path, "r") as f:
        atom_freq = f.readlines()
        for atom in atom_freq:
            atom_symbol, freq = atom.split("\t")
            atom_freqs[atom_symbol] = float(freq)
    scores = {}
    atom_freq_log_inv = {
        atom: 1 / math.log1p(freq)
        for atom, freq in atom_freqs.items()
    }
    min_val = min(atom_freq_log_inv.values())
    for atom, s in atom_freq_log_inv.items():
        scores[atom] = s / min_val

    return scores


def _get_predefined_importances(
    tokenizer: PreTrainedTokenizerBase,
    tokenized_labels: List[List[str]],
    predefined_importances: List[List[float]],
    special_token_importance: float,
) -> List[List[float]]:
    for labels, importances in zip(tokenized_labels, predefined_importances):
        importances.append(special_token_importance)
        for i in range(len(labels)):
            if labels[i] in tokenizer.all_special_tokens:
                importances[i] = special_token_importance
    return predefined_importances


def _pad_list_to_tensor(
    data: List[List[float]],
    max_length: int,
    pad_value: float = -1.0,
) -> torch.Tensor:
    batch_size = len(data)
    padded_tensor = torch.full((batch_size, max_length),
                               pad_value,
                               dtype=torch.float32)
    for i, seq in enumerate(data):
        length = min(len(seq), max_length)
        padded_tensor[i, :length] = torch.tensor(seq[:length],
                                                 dtype=torch.float32)

    return padded_tensor


class Averager:

    def __init__(self, weight: float = 1):
        self.weight = weight
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats):
        for key, value in stats.items():
            self.total[
                key] = self.total[key] * self.weight + value * self.weight
            self.counter[key] = self.counter[key] * self.weight + self.weight

    def average(self):
        averaged_stats = {
            key: tot / self.counter[key]
            for key, tot in self.total.items()
        }
        self.reset()

        return averaged_stats
