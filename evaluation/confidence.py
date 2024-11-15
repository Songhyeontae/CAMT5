from model.loader import Model
from transformers import PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from torch.nn.functional import cross_entropy

import torch
from typing import Tuple, List
from model.representation import Representation

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols
import logging
RDLogger.DisableLog('rdApp.*')
DUMMY_SMILES = "C"

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)

#TODO(hyeontae): Consider better way to calculate perplexity
def get_perplexity(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    device: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    outputs: GenerateOutput,
    length_normalize: float = 1.0,
    representation: Representation = None,
    temperature: float = None,
) -> torch.Tensor:
    assert input_ids.shape == attention_mask.shape
    
    batch_size = input_ids.shape[0]
    max_length = input_ids.shape[1]
    num_sequences = outputs.sequences.shape[0] // batch_size
    best_sequence = outputs.sequences.reshape(batch_size, num_sequences, -1)[:, 0, :]
    
    decoded_labels = tokenizer.batch_decode(best_sequence, skip_special_tokens=True)
    encoded_labels = tokenizer(
        ["<bom>" + label + "<eom>" for label in decoded_labels],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    label_ids = encoded_labels["input_ids"]
    label_attention_mask = encoded_labels["attention_mask"]
    label_ids[label_ids == tokenizer.pad_token_id] = IGNORE_INDEX # Ignore padding token (cross-entropy loss)
    
    with torch.no_grad():
        model_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_ids,
            decoder_attention_mask=label_attention_mask,
        )
        
        logits = model_outputs.logits
        loss_per_token = cross_entropy(
            logits.view(-1, logits.size(-1)), 
            label_ids.view(-1),
            ignore_index=IGNORE_INDEX, 
            reduction="none"
        ).view(batch_size, -1)
        sequence_lengths = label_attention_mask.sum(dim=1)
        length_penalty = (sequence_lengths / (sequence_lengths ** length_normalize))
        
        if temperature is not None:
            importance = _get_importance(
                seq_len=loss_per_token.shape[1],
                sequences=best_sequence,
                temperature=temperature,
                device=device,
                tokenizer=tokenizer,
                representation=representation,
            )
        else:
            importance = torch.ones_like(loss_per_token)
        
        weight = 0.5 * length_penalty.view(-1, 1) + 0.5 * importance
        importance_weighted_loss_per_token = loss_per_token * weight
        loss_per_sample = importance_weighted_loss_per_token.sum(dim=1) / encoded_labels["attention_mask"].sum(dim=1)
        perplexity = torch.exp(loss_per_sample)
        
    return perplexity

def get_entropy(
    model: Model,
    batch_size: int,
    outputs: GenerateOutput,
) -> torch.Tensor:
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True,
    )

    sentence_log_probs = transition_scores.sum(dim=1).reshape(batch_size, -1) # (batch_size, num_sequences)
    
    importance_sampling_weight = torch.nn.functional.softmax(sentence_log_probs/0.01, dim=1)
    entropy = -1 * (sentence_log_probs * importance_sampling_weight).mean(dim=1)
    return entropy

def get_len_norm_entropy(
    model: Model,
    batch_size: int,
    outputs: GenerateOutput,
) -> torch.Tensor:
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True,
    )
    
    pad_token_id = model.config.pad_token_id
    sequence_lengths = (outputs.sequences != pad_token_id).sum(dim=1).reshape(batch_size, -1) # (batch_size, num_sequences)
    sentence_log_probs = transition_scores.sum(dim=1).reshape(batch_size, -1) # (batch_size, num_sequences)
    importance_sampling_weight = torch.nn.functional.softmax(sentence_log_probs/0.01, dim=1)
    entropy = -1 * (sentence_log_probs * importance_sampling_weight / sequence_lengths).mean(dim=1)
    
    return entropy

def get_importance_weighted_entropy(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    representation: Representation,
    device: str,
    batch_size: int,
    outputs: GenerateOutput,
    temperature: float = 0.01,
) -> torch.Tensor:
    sequences = outputs.sequences
    transition_scores = model.compute_transition_scores(
        sequences, outputs.scores, outputs.beam_indices, normalize_logits=True,
    ) # (batch_size * num_sequences, max_len)
    
    importance = _get_importance(
        seq_len=transition_scores.shape[1],
        sequences=sequences,
        temperature=temperature,
        device=device,
        tokenizer=tokenizer,
        representation=representation,
    )

    weighted_transition_scores = transition_scores * importance
    sentence_log_probs = weighted_transition_scores.sum(dim=1).reshape(batch_size, -1) # (batch_size, num_sequences)
    importance_sampling_weight = torch.nn.functional.softmax(sentence_log_probs/0.01, dim=1)
    entropy = -1 * (sentence_log_probs * importance_sampling_weight).mean(dim=1)
    
    return entropy


def get_rdk(
    representation: Representation,
    prediction: List[str],
    target: List[str],
) -> torch.Tensor:
    assert len(prediction) == len(target)
    
    decoded_prediction = [
        representation.decode(pred) for pred in prediction
    ]
    
    rdks = []
    for pred, tar in zip(decoded_prediction, target):
        gen_mol = Chem.MolFromSmiles(pred)
        if gen_mol == None:
            gen_mol = Chem.MolFromSmiles(DUMMY_SMILES)
        target_mol = Chem.MolFromSmiles(tar)
        gen_smiles = Chem.MolToSmiles(gen_mol)

        if gen_smiles == DUMMY_SMILES:
            rdks.append(0)
            continue

        target_fp_rdk = FingerprintMols.GetRDKFingerprint(target_mol)
        gen_fp_rdk = FingerprintMols.GetRDKFingerprint(gen_mol)
        s_rdk = DataStructs.TanimotoSimilarity(gen_fp_rdk, target_fp_rdk)

        rdks.append(s_rdk)
        
    return torch.tensor(rdks)

def get_exact(
    representation: Representation,
    prediction: List[str],
    target: List[str],
) -> torch.Tensor:
    assert len(prediction) == len(target)
    
    decoded_prediction = [
        representation.decode(pred) for pred in prediction
    ]
    
    exacts = []
    for pred, tar in zip(decoded_prediction, target):
        gen_mol = Chem.MolFromSmiles(pred)
        if gen_mol == None:
            gen_mol = Chem.MolFromSmiles(DUMMY_SMILES)
        target_mol = Chem.MolFromSmiles(tar)
        gen_smiles = Chem.MolToSmiles(gen_mol)
        target_smiles = Chem.MolToSmiles(target_mol)

        if gen_smiles == DUMMY_SMILES:
            exacts.append(0)
            continue

        exact_match = (
            1 if \
                Chem.MolToInchi(Chem.MolFromSmiles(gen_smiles)) \
                    == Chem.MolToInchi(Chem.MolFromSmiles(target_smiles)) else 0
        )
        
        exacts.append(exact_match)
        
    return torch.tensor(exacts)

def _get_importance(
    seq_len: int,
    sequences: torch.Tensor,
    temperature: float,
    device: str,
    tokenizer: PreTrainedTokenizer,
    representation: Representation,
) -> torch.Tensor:
    seq_token_sizes = []
    for sequence in sequences:
        token_sizes = []
        for token in sequence:
            if token == tokenizer.pad_token_id:
                continue
            token_mol = tokenizer.decode([token], skip_special_tokens=True)
            size = representation.get_size(token_mol)
            token_sizes.append(size)
        if len(token_sizes) < seq_len:
            token_sizes.extend([-1e9] * (seq_len - len(token_sizes)))
        seq_token_sizes.append(token_sizes)
    seq_token_sizes = torch.tensor(seq_token_sizes).to(device)
    
    importance = torch.nn.functional.softmax(seq_token_sizes/temperature, dim=1)
        
    return importance
