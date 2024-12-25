import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import (CausalLMOutputWithPast,
                                           Seq2SeqLMOutput)

from train.config import ImportanceWeightConfig, Loss

Output = Union[Seq2SeqLMOutput, CausalLMOutputWithPast]

logger = logging.getLogger(__name__)


def get_loss(
    outputs: Output,
    targets: torch.Tensor,
    loss: Loss,
    importance_weight_config: Optional[ImportanceWeightConfig] = None,
    token_importances: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:

    loss_fn = select_loss(loss)(reduction='none')
    logits = outputs.logits
    batch_size, seq_length, num_classes = logits.size()
    logits = logits.view(-1, num_classes)
    targets = targets.view(-1)

    loss = loss_fn(logits, targets)
    loss = loss.view(batch_size, seq_length)

    if importance_weight_config is not None:
        targets = targets.view(batch_size, seq_length)
        assert token_importances is not None, "Token importance must be provided"
        assert token_importances.size() == targets.size()
        weights = get_weights(token_importances, targets,
                              importance_weight_config)
        loss = (loss * weights).sum(dim=-1)  # same as weighted average
    else:
        loss = loss.mean(dim=-1)

    return loss.mean()


def get_weights(
    importances: torch.FloatTensor,
    targets: torch.Tensor,
    importance_weight_config: ImportanceWeightConfig,
) -> torch.FloatTensor:
    mask = targets != -100
    assert torch.all(
        (importances != -1.0) == mask), "Mask and importances must be aligned"
    if importance_weight_config.importance_weighted_loss:
        pass
    elif importance_weight_config.log_importance_weighted_loss:
        importances = torch.log1p(importances)
    else:
        raise ValueError("Invalid loss config")
    importances[~mask] = -1e9

    normalized_importances = nn.functional.softmax(
        importances / importance_weight_config.temperature, dim=-1)
    return normalized_importances


def select_loss(loss: Loss) -> nn.Module:
    if loss == Loss.CE.value:
        return nn.CrossEntropyLoss
    elif loss == Loss.FOCAL.value:
        return FocalLoss
    elif loss == Loss.INVERSE_FOCAL.value:
        return InverseFocalLoss
    else:
        raise ValueError(f"Invalid loss config {loss}")


class FocalLoss(nn.Module):

    def __init__(self, weight=None, gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(((1 - prob)**self.gamma) * log_prob,
                          target_tensor,
                          weight=self.weight,
                          reduction=self.reduction)


class InverseFocalLoss(nn.Module):

    def __init__(self, weight=None, gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(((1 + prob)**self.gamma) * log_prob,
                          target_tensor,
                          weight=self.weight,
                          reduction=self.reduction)
