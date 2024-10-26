from train.config import SchedulerConfig, LRScheduler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR

import torch
import logging
import transformers
import math

logger = logging.getLogger(__name__)

def get_lr_scheduler(optimizer: torch.optim.Optimizer, 
                     total_steps: int,
                     base_lr: int,
                     config: SchedulerConfig) -> torch.optim.lr_scheduler.LRScheduler:
    
    # TODO(hyeontae): Remove the hard-coded schedulers
    if config.name == LRScheduler.COSINE.value:
        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=config.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - config.warmup_steps,
            eta_min=config.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[config.warmup_steps]
        )
    elif config.name == LRScheduler.LEGACY.value:
        msg = "You are using T5 legacy LR Schedule, it's independent from the optim.base_lr"
        logger.info(msg)

        num_steps_optimizer1 = math.ceil(total_steps * 0.9)
        iters_left_for_optimizer2 = total_steps - num_steps_optimizer1

        scheduler1 = LambdaLR(
            optimizer,
            lambda step: min(
                1e-2, 1.0 / math.sqrt(step)
            ) / base_lr if step else 1e-2 / base_lr
        )

        scheduler2 = LinearLR(
            optimizer,
            start_factor=(
                min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / base_lr
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1]
        )
    elif config.name == LRScheduler.CONSTANT.value:

        lr_scheduler = transformers.get_scheduler(
            name=LRScheduler.CONSTANT.value,
            optimizer=optimizer,
        )
    else:
        raise ValueError(f"Invalid LR Scheduler configuration: {config}")

    return lr_scheduler