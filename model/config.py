import dataclasses
from dataclasses import field
from enum import Enum
from typing import List, Optional


class Representation(Enum):
    SMILES = "smiles"
    SELFIES = "selfies"
    FRAG = "frag"


@dataclasses.dataclass
class TokenizerConfig:
    additional_tokens_paths: List[str] = field(default_factory=list)


@dataclasses.dataclass
class MolTokenizerConfig:
    # one of the following should be set
    smiles: Optional[bool] = None
    selfies: Optional[bool] = None
    frag: Optional[bool] = None


@dataclasses.dataclass
class LoadModel:
    from_pretrained: bool = True


@dataclasses.dataclass
class ModelConfig:
    name: str
    load_model: LoadModel
    tokenizer_config: TokenizerConfig
    representation: Representation
    checkpoint_path: Optional[str] = None
    dropout: Optional[float] = field(default=0.1)
