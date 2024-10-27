import re
from typing import List

import torch
import transformers
from torch import nn
from transformers import PreTrainedModel

from train.config import OptimConfig, Optimizer
from train.utils import AdamWScale

NO_DECAY_PARAM_NAME_PATTERN = re.compile("|".join(
    ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]))


def _weight_decay_excluded_params(
        model: PreTrainedModel,
        weight_decay: float) -> List[nn.parameter.Parameter]:
    named_parameters = model.named_parameters()
    parameters = []

    for name, param in named_parameters:
        parameters.append({
            "params":
            param,
            "weight_decay":
            0.0 if NO_DECAY_PARAM_NAME_PATTERN.search(name) else weight_decay,
        })

    return parameters


def get_optimizer(model: PreTrainedModel,
                  optim_confg: OptimConfig) -> torch.optim.Optimizer:
    parameters = _weight_decay_excluded_params(model, optim_confg.weight_decay)

    if optim_confg.name == Optimizer.ADAMW.value:
        optimizer = transformers.AdamW(
            parameters,
            lr=optim_confg.base_lr,
        )
    elif optim_confg.name == Optimizer.ADAMWSCALE.value:
        optimizer = AdamWScale(
            parameters,
            lr=optim_confg.base_lr,
        )
    elif optim_confg.name == Optimizer.ADAFACTOR.value:
        from transformers import Adafactor
        optimizer = Adafactor(
            parameters,
            lr=optim_confg.base_lr,
            relative_step=False,
        )
    else:
        raise ValueError(f"Invalid optimizer configuration: {optim_confg}")

    return optimizer
