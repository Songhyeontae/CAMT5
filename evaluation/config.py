import dataclasses
from dataclasses import field
from enum import Enum
from typing import Optional, Dict

class Device(Enum):
    CPU = "cpu"
    GPU = "gpu"
    
class Confidence(Enum):
    NEG_PERPLEXITY = "neg_perplexity"
    NEG_NORMALIZED_PERPLEXITY = "neg_normalized_perplexity"
    PROBABILITY = "probability"
    ENTROPY = "entropy"
    LEN_NORM_ENTROPY = "len_norm_entropy"
    IMPORTANCE_WEIGHTED_ENTROPY = "importance_weighted_entropy"
    IMPORTANCE_WEIGHTED_PERPLEXITY = "importance_weighted_perplexity"
    ORACLE_RDK = "oracle_RDK"
    ORACLE_EXACT = "oracle_exact"
    
@dataclasses.dataclass
class DataConfig:
    data_path: str
    batch_size: int
    chunk_size: Optional[int] = 1024
    num_workers: Optional[int] = 8
    prefetch_factor: Optional[int] = 2
    
@dataclasses.dataclass
class ConfidenceConfig:
    confidence: Confidence
    length_normalize: Optional[float] = 1.0
    temperature: Optional[float] = 0.01 # For importance weighted entropy

@dataclasses.dataclass
class PredictConfig:
    max_length: int
    num_beams: int
    confidence_config: ConfidenceConfig
    num_return_sequences: Optional[int] = 1
    cache_paths: Dict[str, str] = field(default_factory=dict)

@dataclasses.dataclass
class EvalConfig:
    data_config: DataConfig
    predict_config: PredictConfig
    device: Device
    ensemble: Optional[bool] = False

    