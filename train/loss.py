from train.config import LossConfig
from transformers.modeling_outputs import (CausalLMOutputWithPast,
                                           Seq2SeqLMOutput)

from typing import Dict, Union
import torch
import logging

Output = Union[Seq2SeqLMOutput, CausalLMOutputWithPast]

logger = logging.getLogger(__name__)

def get_loss(
    outputs: Output,
    targets: torch.Tensor,
    loss_config: LossConfig,
    token_importance: torch.FloatTensor,
) -> torch.FloatTensor:
    if not loss_config:
        return outputs.loss

    logits = outputs.logits
    token_importances = token_importance[targets]
    token_importances[targets==-100] = -1 # ignore index
    
    weights = get_weights(token_importances, loss_config)
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    batch_size, seq_length, num_classes = logits.size()
    logits = logits.view(-1, num_classes)
    targets = targets.view(-1)

    loss = loss_fn(logits, targets)
    loss = loss.view(batch_size, seq_length)
    weighted_loss = (loss * weights).sum(dim=-1).mean() # same as weighted average
    
    return weighted_loss
    
def get_weights(
    importances: torch.FloatTensor,
    loss_config: LossConfig,
) -> torch.FloatTensor:
    mask = importances != -1
    if loss_config.importance_weighted_loss:
        pass
    elif loss_config.log_importance_weighted_loss:
        importances = torch.log1p(importances)
    else:
        raise ValueError("Invalid loss config")
    
    importances[~mask] = -1e9
    normalized_importances = torch.nn.functional.softmax(importances / loss_config.temperature, dim=-1)
    return normalized_importances