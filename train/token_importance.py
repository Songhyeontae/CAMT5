import math
from functools import lru_cache
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizerBase

from model.representation import Representation
from train.config import TokenImportance, TokenImportanceConfig
from utils import to_absolute_path


def get_token_importance(
    config: TokenImportanceConfig,
    tokenized_labels: List[List[str]],
    tokenizer: PreTrainedTokenizerBase,
    representation: Representation,
    predefined_importances: Optional[List[List[float]]] = None,
) -> List[List[float]]:

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
    elif config.token_importance == TokenImportance.MOTIF_FREQ.value:
        token_importances = _get_motif_freq_importances(
            tokenized_labels,
            tokenizer,
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

def _get_motif_freq_importances(
    tokenized_labels: List[List[str]],
    tokenizer: PreTrainedTokenizerBase,
    motif_freq_path: str,
    special_token_importance: float,
) -> List[List[float]]:
    motif_freq_scores = _get_motif_freq_score(motif_freq_path)
    token_importances = []
    for label in tokenized_labels:
        token_importance = []
        for token in label:
            if token in tokenizer.all_special_tokens:
                token_importance.append(special_token_importance)
            else:
                token_importance.append(motif_freq_scores.get(token, special_token_importance))
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

@lru_cache(maxsize=1)
def _get_motif_freq_score(motif_freq_path: str) -> Dict[str, float]:
    if motif_freq_path is None:
        raise ValueError("Please provide motif frequency path")

    motif_freq_path = to_absolute_path(motif_freq_path)
    motif_freqs = {}
    with open(motif_freq_path, "r") as f:
        motif_freq = f.readlines()
        for motif in motif_freq:
            motif_symbol, freq = motif.split("\t")
            motif_freqs[motif_symbol] = float(freq)
    scores = {}
    motif_freq_log_inv = {
        motif: 1 / math.log1p(freq)
        for motif, freq in motif_freqs.items()
    }
    min_val = min(motif_freq_log_inv.values())
    for motif, s in motif_freq_log_inv.items():
        scores[motif] = s / min_val

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
