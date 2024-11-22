import dataclasses
from dataclasses import field
from enum import Enum
from typing import Optional


class TestTask(Enum):
    MOL2TEXT = "mol2text"
    TEXT2MOL = "text2mol"
    TEXT2FRAG = "text2frag"
    DTI = "dti"
    PEER = "peer"
    MOLNET = "molnet"


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


@dataclasses.dataclass
class EvalConfig:
    task: TestTask
    every_steps: int
    total_steps: Optional[int] = None
    eval_results_path: Optional[str] = None
    tensorboard_path: Optional[str] = None


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
    grad_clip: float
    lr_scheduler_config: Optional[SchedulerConfig] = field(
        default_factory=SchedulerConfig)


@dataclasses.dataclass
class TemperatureSchedulerConfig:
    decay: TemperatureDecay
    decay_rate: float
    every_steps: int
    min_temperature: Optional[float] = None


@dataclasses.dataclass
class LossConfig:
    temperature: Optional[float] = None
    temperature_scheduler_config: Optional[TemperatureSchedulerConfig] = None
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

    loss_config: Optional[LossConfig] = field(default_factory=LossConfig)
    eval_config: Optional[EvalConfig] = field(default_factory=EvalConfig)
    test_bsz_multi: Optional[int] = 1
    do_compile: Optional[bool] = False


@dataclasses.dataclass
class DataConfig:
    num_workers: int
    max_seq_len: int
    max_target_len: int
    add_task_name: bool
    add_task_definition: bool
    num_pos_examples: int
    num_neg_examples: int
    add_explanation: bool
    tk_instruct: bool

    input_length: Optional[int] = None
    mlm_probability: Optional[float] = None
    mean_noise_span_length: Optional[float] = None

    # config for finetuning dataset
    exec_file_path: Optional[str] = None
    data_dir: Optional[str] = None
    task_dir: Optional[str] = None
    max_num_instances_per_task: Optional[int] = None
    max_num_instances_per_eval_task: Optional[int] = None
