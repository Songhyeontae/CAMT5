import dataclasses
from dataclasses import field
from enum import Enum
from typing import Any, Optional


class TestTask(Enum):
    MOL2TEXT = "mol2text"
    TEXT2MOL = "text2mol"
    TEXT2FRAG = "text2frag"
    DTI = "dti"
    PEER = "peer"
    MOLNET = "molnet"
    MLM = "mlm"


class TokenImportance(Enum):
    ATOM_COUNT = "atom_count"
    ATOM_FREQ = "atom_freq"
    PREDEFINED = "predefined"


class Device(Enum):
    CPU = "cpu"
    GPU = "gpu"


class Optimizer(Enum):
    ADAMW = "adamw"
    ADAMWSCALE = "adamwscale"
    ADAFACTOR = "adafactor"


class LRScheduler(Enum):
    COSINE = "cosine"
    LEGACY = "legacy"
    CONSTANT = "constant"


class TemperatureDecay(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class Loss(Enum):
    CE = "cross_entropy"
    FOCAL = "focal"
    INVERSE_FOCAL = "inverse_focal"


@dataclasses.dataclass
class Wandb:
    project: str
    run_name: str


@dataclasses.dataclass
class EvalConfig:
    task: TestTask

    # If this field is not None, the model will be evaluated during training
    every_steps: Optional[int] = None

    max_target_len: Optional[int] = None
    total_steps: Optional[int] = None
    eval_results_path: Optional[str] = None
    tensorboard_path: Optional[str] = None
    wandb: Optional[Wandb] = None


@dataclasses.dataclass
class Checkpoint:
    every_steps: int
    path: Optional[str] = None


@dataclasses.dataclass
class LoggingConfig:
    every_steps: int

    # metrics
    accuracy: bool
    grad_l2: bool
    weights_l2: bool


@dataclasses.dataclass
class SchedulerConfig:
    name: LRScheduler
    warmup_steps: Optional[int] = None
    # for cosine
    final_cosine: Optional[float] = field(default=0.0)


@dataclasses.dataclass
class OptimConfig:
    name: Optimizer
    weight_decay: float
    base_lr: float
    total_steps: int
    grad_clip: Optional[float] = field(default=-1.0)
    lr_scheduler_config: Optional[SchedulerConfig] = field(
        default_factory=SchedulerConfig)


@dataclasses.dataclass
class TemperatureSchedulerConfig:
    decay: TemperatureDecay
    decay_rate: float
    every_steps: int
    min_temperature: Optional[float] = None


@dataclasses.dataclass
class TokenImportanceConfig:
    token_importance: TokenImportance
    special_token_importance: Optional[float] = 1.0
    # path to the atom count file
    atom_freq_path: Optional[str] = None


@dataclasses.dataclass
class ImportanceWeightConfig:
    temperature: float
    temperature_scheduler_config: Optional[TemperatureSchedulerConfig] = None

    # importance weight or log importance weight
    importance_weighted_loss: Optional[bool] = False
    log_importance_weighted_loss: Optional[bool] = False


@dataclasses.dataclass
class TrainConfig:
    device: Device
    seed: int
    shot: int
    batch_size: int
    grad_acc: int
    optim_config: OptimConfig

    checkpoint: Checkpoint
    logging_config: LoggingConfig

    importance_weight_config: Optional[ImportanceWeightConfig] = None
    loss: Optional[Loss] = field(default=Loss.CE.value)
    eval_config: Optional[EvalConfig] = field(default_factory=EvalConfig)
    test_bsz_multi: Optional[int] = 1
    do_compile: Optional[bool] = False


@dataclasses.dataclass
class Text2MolDataConfig:
    max_seq_len: int
    max_target_len: int
    exec_file_path: str
    data_dir: str
    task_dir: str
    max_num_instances_per_task: Optional[int] = None
    max_num_instances_per_eval_task: Optional[int] = None


# Config for MLM Task. Use C4, zinc dataset.
@dataclasses.dataclass
class MLMDataConfig:
    mlm_probability: float
    mean_noise_span_length: int
    input_length: int


@dataclasses.dataclass
class DataConfig:
    num_workers: int

    text_2_mol_data_config: Optional[Text2MolDataConfig] = None
    mlm_data_config: Optional[MLMDataConfig] = None
    token_importance_config: Optional[TokenImportanceConfig] = None
