import random
import string

from rdkit import RDLogger
from rdkit.Chem.Fingerprints import FingerprintMols
from transformers.data.data_collator import *

from metrics.text2mol_metrics import get_rdk_metric
from model.representation import Representation

RDLogger.DisableLog('rdApp.*')

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from train.config import TokenImportanceConfig
from utils import to_absolute_path


@dataclass
class DataCollatorForNI:
    tokenizer: PreTrainedTokenizerBase
    representation: Representation
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    token_importance_config: TokenImportanceConfig = None
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool = False

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 0,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # example only
                    {
                        "add_task_name": False,
                        "add_task_definition": False,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # instruction + pos examples
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # instruction + pos examples + neg examples
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 2,
                        "add_explanation": False,
                    },
                    # instruction + pos (w. explanation)
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": True,
                    },
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "

            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][
                        0].strip()
                else:
                    definition = "Definition: " + instance["Definition"].strip(
                    )
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"

            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(
                    instance["Positive Examples"][:num_pos_examples]):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += (
                        f" Explanation: {pos_example['explanation'].strip()}")
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if (len(
                        self.tokenizer(definition + " ".join(pos_examples) +
                                       pos_example_str +
                                       task_input)["input_ids"]) <=
                        self.max_source_length):
                    pos_examples.append(pos_example_str)
                else:
                    break

            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(
                    instance["Negative Examples"][:num_neg_examples]):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += (
                        f" Explanation: {neg_example['explanation'].strip()}")
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if (len(
                        self.tokenizer(definition + " ".join(pos_examples) +
                                       " ".join(neg_examples) +
                                       neg_example_str +
                                       task_input)["input_ids"]) <=
                        self.max_source_length):
                    neg_examples.append(neg_example_str)
                else:
                    break

            source = (task_name + definition + "".join(pos_examples) +
                      "".join(neg_examples) + task_input)
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(
                    self.tokenizer.decode(
                        tokenized_source[:self.max_source_length],
                        skip_special_tokens=True,
                    ))

        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            importance = []
            if "importance" in batch[0]["Instance"]:
                importance = [
                    ex["Instance"]["importance"][0] for ex in batch
                ]
            if self.token_importance_config is not None:
                importance_scores = self._get_token_importance(labels, importance)

            if self.text_only:
                model_inputs["labels"] = labels
            else:
                labels = self.tokenizer(
                    labels,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                )

                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(
                    ~label_mask, self.label_pad_token_id)

            if self.token_importance_config is not None:
                model_inputs["token_importance"] = pad_sequence_to_length(
                    importance_scores,
                    batch_first=True,
                    padding_value=-1,
                    desired_length=model_inputs["labels"].shape[1])
        else:
            model_inputs["labels"] = None
        return model_inputs

    def _get_token_importance(
        self,
        label_texts: List[str],
        importances: List[List[float]]
    ) -> List[torch.Tensor]:
        """
        Calculates importance of each output token by removing one at a time.

        Args:
            label_texts (list[str]): Original label texts.

        Returns:
            list[list[float]]: Importance scores for each token.
        """
        config = self.token_importance_config

        if config.sim_base_importance:
            return [torch.Tensor(importance) for importance in importances]
        elif config.atom_count_importance:
            return _get_atom_count_based_importance(self.tokenizer,
                                                    self.representation,
                                                    label_texts)
        elif config.atom_freq_score_importance:
            return _get_atom_freq_score_based_importance(
                self.tokenizer, self.representation, config.atom_freq_path,
                label_texts)
        else:
            raise ValueError("Invalid importance configuration")

def _get_atom_count_based_importance(
    tokenizer: PreTrainedTokenizerBase,
    representation: Representation,
    label_texts: List[str],
):
    all_importance_scores = []
    for label_text in label_texts:
        importance_scores = []  # Store importance scores for the current label
        tokenized_label = tokenizer.tokenize(label_text)

        for token in tokenized_label:
            atom_count = representation.get_size(token)
            importance_scores.append(atom_count)
        all_importance_scores.append(torch.Tensor(importance_scores))

    return all_importance_scores


def _get_atom_freq_score_based_importance(
    tokenizer: PreTrainedTokenizerBase,
    representation: Representation,
    atom_freq_path: str,
    label_texts: List[str],
):
    all_token_importance_scores = []
    for label_text in label_texts:
        importance_scores = []
        tokenized_label = tokenizer.tokenize(label_text)

        for token in tokenized_label:
            atom_score = _get_atom_score(atom_freq_path)
            atom_freq = representation.get_atom_weighted_score(
                token, atom_score)
            importance_scores.append(atom_freq)
        all_token_importance_scores.append(torch.Tensor(importance_scores))

    return all_token_importance_scores


def _get_atom_score(atom_freq_path: str) -> Dict[str, float]:
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


def pad_sequence_to_length(sequences,
                           desired_length,
                           padding_value=0,
                           batch_first=True):
    """
    Pads or truncates sequences to the desired length.

    Args:
        sequences (list of torch.Tensor): A list of 1D tensors to be padded or truncated.
        desired_length (int): The target length for each sequence.
        padding_value (int, optional): The value used for padding. Default is 0.
        batch_first (bool, optional): If True, returns batch as (batch_size, seq_length).
                                      If False, returns batch as (seq_length, batch_size).

    Returns:
        torch.Tensor: A tensor with sequences padded or truncated to the desired length.
    """
    padded_sequences = []
    for seq in sequences:
        # Truncate if the sequence is longer than desired_length
        if len(seq) > desired_length:
            padded_seq = seq[:desired_length]
        # Pad if the sequence is shorter than desired_length
        else:
            padding_needed = desired_length - len(seq)
            padded_seq = torch.cat(
                [seq, torch.full((padding_needed, ), padding_value)])

        padded_sequences.append(padded_seq)

    # Stack the padded sequences into a tensor
    if batch_first:
        return torch.stack(padded_sequences)
    else:
        return torch.stack(padded_sequences).transpose(0, 1)
