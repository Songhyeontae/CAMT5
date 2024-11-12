import logging
import dataclasses
import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union
from itertools import islice
from collections import defaultdict
from itertools import islice

from core.task import BaseTaskCls
from evaluation.config import EvalConfig, PredictConfig, Device, Confidence
from model.config import ModelConfig
from model.loader import ModelLoader, Model
from model.representation import Representation
from metrics.text2mol_metrics import get_text2mol_metrics

from transformers import PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from evaluation.dataloader import get_dataloader
from evaluation.confidence import get_perplexity, get_entropy, get_len_norm_entropy, get_importance_weighted_entropy, get_rdk, get_exact

from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class MolLM:
    model: Model
    tokenizer: PreTrainedTokenizer
    representation: Representation
    
class App(BaseTaskCls):

    def __init__(
        self,
        eval_config: EvalConfig,
        model_configs: List[ModelConfig],
        **kwargs,
    ):
        super(App, self).__init__(**kwargs)
        self.eval_config = eval_config
        self.model_configs = model_configs

    def run(self, **kwargs):
        del kwargs
        _validate_eval_config(self.eval_config, self.model_configs)
        mol_lms = {}
        for model_config in self.model_configs:
            model_loader = ModelLoader(model_config)
            mol_lm = MolLM(
                model=model_loader.load_model(),
                tokenizer=model_loader.load_tokenizer(),
                representation=model_loader.load_representation()
            )
            mol_lms[model_config.representation_type] = mol_lm
            
        evaluate(self.eval_config, mol_lms)
        

def evaluate(
    config: EvalConfig,
    mol_lms: Dict[str, MolLM]
):
    if config.device == Device.CPU.value:
        device = "cpu"
    elif config.device == Device.GPU.value:
        device = "cuda"
    else:
        raise ValueError(f"Invalid device: {config.device}")

    dataloader = get_dataloader(
        data_config=config.data_config,
    )
    total_steps = len(dataloader)
    predictions_per_model = defaultdict(list)
    confidences_per_model = defaultdict(list)
    targets = []
    
    cache_paths = (config.predict_config.cache_paths
    if config.predict_config.cache_paths else {})
    for rep_name, cache_path in cache_paths.items():
        if rep_name not in mol_lms:
            continue
        logger.info("Loading %s model from '%s'...", rep_name, cache_path)
        predictions, confidences = _load_cache(cache_path)
        predictions_per_model[rep_name] = predictions
        confidences_per_model[rep_name] = confidences

    assert predictions_per_model.keys() == confidences_per_model.keys(), (
        "Predictions and confidences should have the same keys"
    )
    
    cached_keys = set(predictions_per_model.keys())
    non_cached_mol_lms = {
        rep_name: mol_lm for rep_name, mol_lm 
        in mol_lms.items() if rep_name not in cached_keys
    }
    for batch in tqdm(dataloader, total=total_steps):
        target, description = batch
        targets.extend(target)
        for rep_name, mol_lm in non_cached_mol_lms.items():
            predictions, confidences = predict(
                mol_lm=mol_lm,
                description=description,
                device=device,
                config=config.predict_config,
                target=target,
            )
            predictions_per_model[rep_name].extend(predictions)
            confidences_per_model[rep_name].extend(confidences)
   
    for rep_name, mol_lm in non_cached_mol_lms.items():
        _cache_predictions(
            key=rep_name,
            predictions=predictions_per_model[rep_name],
            confidences=confidences_per_model[rep_name],
        )
    
    decoded_ensembled_predictions = enemble_and_decode_predictions(
        mol_lms=mol_lms,
        predictions_per_model=predictions_per_model,
        confidences_per_model=confidences_per_model,
    )
    
    metrics = get_text2mol_metrics(
        predictions=decoded_ensembled_predictions,
        references=targets,
    )
    
    logger.info("Metrics: %s", metrics)

def predict(
    mol_lm: MolLM,
    description: Tuple[str],
    device: str,
    config: PredictConfig,
    target: Tuple[str] = None,
) -> Tuple[List[str], List[float]]:
    mol_lm.model.to(device)
    mol_lm.model.eval()

    batch_size = len(description)
    with torch.no_grad():
        inputs = mol_lm.tokenizer(
            description,
            max_length=config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        
        outputs = mol_lm.model.generate(
            input_ids=inputs["input_ids"],
            generation_config=mol_lm.model.generation_config,
            max_length=config.max_length,
            num_beams=config.num_beams,
            num_return_sequences=config.num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
        )

        sequences = outputs.sequences.reshape(batch_size, config.num_return_sequences, -1)
        best_sequences = sequences[:, 0, :] # Sequences are sorted by score
        decoded_sequences = mol_lm.tokenizer.batch_decode(
            best_sequences, skip_special_tokens=True)
        
        #TODO: calculate confidence
        confidences = get_confidence(
            mol_lm=mol_lm,
            device=device,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            outputs=outputs, 
            confidence=config.confidence_config.confidence,
            length_normalize=config.confidence_config.length_normalize,
            # for oracle confidence
            prediction=decoded_sequences,
            target=target,
        )
        
    return decoded_sequences, confidences

def get_confidence(
    mol_lm: MolLM,
    device: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    outputs: Union[GenerateOutput, torch.LongTensor],
    confidence: Confidence,
    length_normalize: float = 1.0,
    temperature: float = 0.01,
    prediction: List[str] = None,
    target: List[str] = None,
) -> List[float]:
    
    if confidence == Confidence.NEG_PERPLEXITY.value:
        perplexity = get_perplexity(
            model=mol_lm.model,
            tokenizer=mol_lm.tokenizer,
            device=device,
            input_ids=input_ids,
            attention_mask=attention_mask,
            outputs=outputs,
        )

        return [-perp for perp in perplexity.tolist()]
    elif confidence == Confidence.NEG_NORMALIZED_PERPLEXITY.value:
        perplexity = get_perplexity(
            model=mol_lm.model,
            tokenizer=mol_lm.tokenizer,
            device=device,
            input_ids=input_ids,
            attention_mask=attention_mask,
            outputs=outputs,
            length_normalize=length_normalize,
        )
        return [-perp for perp in perplexity.tolist()]
    elif confidence == Confidence.PROBABILITY.value:
        pass
    elif confidence == Confidence.ENTROPY.value:
        entropy = get_entropy(
            model=mol_lm.model,
            batch_size=input_ids.shape[0],
            outputs=outputs,
        )
        return [-ent for ent in entropy.tolist()]
    elif confidence == Confidence.LEN_NORM_ENTROPY.value:
        entropy = get_len_norm_entropy(
            model=mol_lm.model,
            batch_size=input_ids.shape[0],
            outputs=outputs,
        )
        
        return [-ent for ent in entropy.tolist()]
    elif confidence == Confidence.IMPORTANCE_WEIGHTED_ENTROPY.value:
        entropy = get_importance_weighted_entropy(
            model=mol_lm.model,
            tokenizer=mol_lm.tokenizer,
            representation=mol_lm.representation,
            device=device,
            batch_size=input_ids.shape[0],
            outputs=outputs,
            temperature=temperature,
        )
        return [-ent for ent in entropy]
    
    elif confidence == Confidence.ORACLE_RDK.value:
        assert target is not None, "Target should be provided for oracle confidence"
        assert prediction is not None, "Prediction should be provided for oracle confidence"
        
        rdk = get_rdk(
            representation=mol_lm.representation,
            prediction=prediction,
            target=target,
        )
        return rdk.tolist() 
    elif confidence == Confidence.ORACLE_EXACT.value:
        assert target is not None, "Target should be provided for oracle confidence"
        assert prediction is not None, "Prediction should be provided for oracle confidence"
        
        exact = get_exact(
            representation=mol_lm.representation,
            prediction=prediction,
            target=target,
        )
        return exact.tolist()    
    else:
        raise ValueError(f"Invalid confidence: {confidence}")

def enemble_and_decode_predictions(
    mol_lms: Dict[str, MolLM],
    predictions_per_model: Dict[str, List[str]],
    confidences_per_model: Dict[str, List[float]],
) -> List[str]:
    
    logger.info("Ensembling predictions for model representations: %s", list(mol_lms.keys()))
    assert len(predictions_per_model) == len(confidences_per_model), (
        "Predictions and confidences should have the same keys"
    )
    
    num_predictions = len(predictions_per_model[list(mol_lms.keys())[0]])
    assert all(
        len(predictions) == num_predictions
        for predictions in predictions_per_model.values()
    ), "Predictions should have the same length"
    assert all(
        len(confidences) == num_predictions
        for confidences in confidences_per_model.values()
    ), "Confidences should have the same length"
    
    ensembled_predictions = []
    model_win_counts = defaultdict(int)
    for i in range(num_predictions):
        model_confidences = {}
        for rep_name, confidences in confidences_per_model.items():
            model_confidences[rep_name] = confidences[i]
        confident_rep = max(model_confidences, key=model_confidences.get)
        decoded_prediction = mol_lms[confident_rep].representation.decode(
            predictions_per_model[confident_rep][i]
        )
        model_win_counts[confident_rep] += 1
        ensembled_predictions.append(decoded_prediction)
    
    logger.info("Model win counts: %s", model_win_counts)
    return ensembled_predictions
                  
def _validate_eval_config(eval_config: EvalConfig, model_configs: List[ModelConfig]):
    assert eval_config.predict_config.num_beams >= eval_config.predict_config.num_return_sequences, (
        "num_beams should be greater than num_return_sequences"
    )
    assert not (eval_config.ensemble ^ (len(model_configs) > 1)), (
        "Multiple models should be used for ensemble evaluation"
    )
    
    rep_types = [model_config.representation_type for model_config in model_configs]
    assert len(set(rep_types)) == len(rep_types), (
        "Models should have different representation types"
    )

def _cache_predictions(
    key: str,
    predictions: List[str],
    confidences: List[float],
) -> None:
    """
    Caches predictions and logs the file path.

    Args:
        key (str): The identifier for the predictions file.
        predictions (List[str]): List of prediction strings.
        confidences (List[float]): List of confidence scores corresponding to each prediction.
    """
    assert len(predictions) == len(confidences), (
        "Predictions and confidences should have the same length."
    )

    file_path = f"{key}_predictions.txt"
    with open(file_path, "w") as f:
        for prediction, confidence in zip(predictions, confidences):
            f.write(f"{prediction}\t({confidence:.6f})\n")
    
    abs_path = os.path.abspath(file_path)
    logger.info("Predictions for '%s' are cached to '%s'.", key, abs_path)
    
def _load_cache(
    cache_path: str,
) -> Tuple[List[str], List[float]]:
    """
    Loads cached predictions and confidences from a file.

    Args:
        cache_path (str): The path to the cache file.

    Returns:
        Tuple[List[str], List[float]]: A tuple of predictions and confidences.
    """
    predictions = []
    confidences = []
    with open(cache_path, "r") as f:
        for line in f:
            prediction, confidence = line.strip().split("\t")
            predictions.append(prediction)
            confidences.append(float(confidence[1:-1])) # Remove parentheses
    
    return predictions, confidences
    