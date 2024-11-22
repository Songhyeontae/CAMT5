import dataclasses
import logging
import math
import os
import time
from itertools import islice
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import torch
from accelerate import Accelerator
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import (CausalLMOutputWithPast,
                                           Seq2SeqLMOutput)

from metrics.text2mol_metrics import get_text2mol_metrics
from model.loader import Model
from model.representation import Representation, Selfies, get_importance
from train.config import (DataConfig, Device, LossConfig, TemperatureDecay,
                          TestTask, TrainConfig)
from train.loss import get_loss
from train.optimizer import get_optimizer
from train.scheduler import get_lr_scheduler
from train.utils import Averager
from utils import to_absolute_path

Output = Union[Seq2SeqLMOutput, CausalLMOutputWithPast]
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CurrentState:
    train_step: int
    train_epoch: int
    last_log: float


@dataclasses.dataclass
class GenerationResult:
    predictions: torch.Tensor
    scores: Optional[torch.Tensor] = None


class Trainer:

    def __init__(self, config: TrainConfig, data_config: DataConfig):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.config = config
        self.data_config = data_config
        # Averager for logging
        self.average_logger = Averager()
        self.tensorboard_writer = SummaryWriter(
            log_dir=to_absolute_path(config.eval_config.tensorboard_path))

    def __del__(self):
        self.tensorboard_writer.close()

    def train(
        self,
        model: Model,
        tokenizer: PreTrainedTokenizer,
        representation: Representation,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
    ):
        accelerator = Accelerator(
            cpu=(self.config.device == Device.CPU.value), )

        logging.info(f"Using {accelerator.device}")

        optim_config = self.config.optim_config
        optimizer = get_optimizer(model, optim_config)

        lr_scheduler_config = optim_config.lr_scheduler_config
        lr_scheduler = get_lr_scheduler(optimizer, optim_config.total_steps,
                                        optim_config.base_lr,
                                        lr_scheduler_config)

        # Prepare distributed training, mixed precision, etc.
        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader)

        if self.config.do_compile:
            torch.compile(model)

        # Set the initial state of the training
        self.current_state = CurrentState(
            train_step=1,
            train_epoch=1,
            last_log=time.time(),
        )

        self._train(
            model=model,
            tokenizer=tokenizer,
            representation=representation,
            train_dataloader=train_dataloader,
            validation_dataloader=eval_dataloader,
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        self.evaluate(
            dataloader=test_dataloader,
            model=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            representation=representation,
            accelerator=accelerator,
            prefix="test",
        )

    def evaluate(
        self,
        dataloader: DataLoader,
        model: Model,
        tokenizer: PreTrainedTokenizer,
        representation: Representation,
        accelerator: Accelerator,
        prefix: str = "validation",
    ):
        # Set Model to eval mode
        model.eval()

        eval_config = self.config.eval_config
        test_task = eval_config.task
        total_steps = len(dataloader)
        if eval_config.total_steps:
            total_steps = min(eval_config.total_steps, total_steps)

        metric = TaskHelper.set_task_metrics(test_task)

        def decode(preds: torch.Tensor) -> List[str]:
            #TODO(hyeontae): Check the logic
            preds[preds == -100] = tokenizer.pad_token_id
            decoded_preds = tokenizer.batch_decode(
                preds,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True)
            preds = [pred.replace(" ", "").strip() for pred in decoded_preds]
            return preds

        input_total, reference_total, prediction_total = [], [], []
        samples_seen = 0

        for step, batch in tqdm(enumerate(islice(dataloader, total_steps)),
                                total=total_steps):
            batch = batch.to(accelerator.device)
            generation_result = TaskHelper.generate_results(
                test_task, model, self.data_config, batch)

            decoded_inputs = decode(batch["input_ids"])
            decoded_references = decode(batch["labels"])
            decoded_predictions = decode(generation_result.predictions)

            parsed_inputs, parsed_predictions, parsed_references =\
                TaskHelper.parse(
                    test_task,
                    representation,
                    decoded_inputs,
                    decoded_predictions,
                    decoded_references,
                    generation_result.scores,
                )

            # If we are in a multiprocess environment, the last batch has duplicates
            if step == len(dataloader) - 1:
                parsed_predictions = parsed_predictions[:len(dataloader.dataset
                                                             ) - samples_seen]
                parsed_references = parsed_references[:len(dataloader.dataset
                                                           ) - samples_seen]
            else:
                samples_seen += len(parsed_references)

            # Update metrics
            metric.add_batch(
                predictions=parsed_predictions,
                references=[(parsed_references[i], parsed_inputs[i])
                            for i in range(len(parsed_references))],
            )

            input_total.extend(parsed_inputs)
            reference_total.extend(parsed_references)
            prediction_total.extend(parsed_predictions)

        eval_metric = metric.compute()

        #TODO(hyeontae): Implement Custom Metrics (RDK, MACCS, Morgan, etc.)
        eval_metric.update(
            TaskHelper.set_additional_metrics(
                test_task,
                prediction_total,
                reference_total,
            ))

        for k, v in eval_metric.items():
            self.tensorboard_writer.add_scalar(f"{prefix}/{k}", v,
                                               self.current_state.train_step)

        self._log_stats(eval_metric, prefix=prefix)
        if eval_config.eval_results_path is not None:
            self._write_eval_results(prefix, eval_metric)

        # Set Model to train mode
        model.train()

    def _train(
        self,
        model: Model,
        tokenizer: PreTrainedTokenizer,
        representation: Representation,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader],
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        # Set Model to train mode
        model.train()

        current_state = self.current_state
        optim_config = self.config.optim_config

        # for token_weighted loss
        token_importance = get_token_importance(
            tokenizer=tokenizer,
            representation=representation,
        ).to(accelerator.device)

        # Start training loop
        while current_state.train_step <= optim_config.total_steps:
            train_dataset = train_dataloader.dataset
            if isinstance(train_dataset, IterableDataset):
                train_dataset.set_epoch(current_state.train_epoch)

            # In case there is a remainder from previous epoch, we need to reset the optimizer
            optimizer.zero_grad(set_to_none=True)

            for batch_id, batch in enumerate(train_dataloader, start=1):
                if current_state.train_step > optim_config.total_steps:
                    break

                outputs: Output = model(**batch)
                loss = get_loss(
                    outputs=outputs,
                    targets=batch["labels"],
                    loss_config=self.config.loss_config,
                    token_importance=token_importance,
                )
                self.average_logger.update(
                    {'loss': loss.detach().float().item()})
                accelerator.backward(loss / self.config.grad_acc)

                if batch_id % self.config.grad_acc == 0:
                    self._update_metrics(model, batch, outputs)

                    if self.config.optim_config.grad_clip > 0:
                        # clip grad norm
                        accelerator.clip_grad_norm_(
                            parameters=model.parameters(),
                            max_norm=self.config.optim_config.grad_clip,
                            norm_type=2,
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    update_temperature(self.config.loss_config,
                                       current_state.train_step)
                    # log hyperparameters
                    lr = optimizer.param_groups[0]['lr']
                    self.average_logger.update({'lr': lr})
                    if self.config.loss_config.temperature is not None:
                        temperature = self.config.loss_config.temperature
                        self.average_logger.update(
                            {'temperature:': temperature})

                    # reset gradients
                    optimizer.zero_grad(set_to_none=True)

                    # log metrics, hyperparameters
                    self._maybe_log_metrics()

                    # evaluate and save checkpoint
                    if accelerator.is_main_process:
                        if self.config.eval_config != None:
                            self._maybe_validate(
                                dataloader=validation_dataloader,
                                model=accelerator.unwrap_model(model),
                                tokenizer=tokenizer,
                                representation=representation,
                                accelerator=accelerator,
                            )

                        self._maybe_save_checkpoint(accelerator, )

                    accelerator.wait_for_everyone()
                    current_state.train_step += 1
            current_state.train_epoch += 1

    def _update_metrics(self, model: Model, batch: Dict[str, torch.Tensor],
                        outputs: Output):
        metrics = {}

        # TODO(hyeontae): Remove hard-coded metrics
        if self.config.logging_config.accuracy:
            correct = (
                outputs.logits.argmax(-1) == batch["labels"]).sum().item()
            accuracy = correct / batch["labels"].numel()
            metrics['accuracy'] = accuracy

        if self.config.logging_config.grad_l2:
            grad_l2 = (sum(p.grad.detach().data.norm(2).item()**2
                           for p in model.parameters())**0.5)
            metrics['grad_l2'] = grad_l2

        if self.config.logging_config.weights_l2:
            weights_l2 = sum(p.detach().norm(2).item()**2
                             for p in model.parameters())**0.5
            metrics['weights_l2'] = weights_l2

        self.average_logger.update(metrics)

    def _maybe_log_metrics(self):
        if self.current_state.train_step % self.config.logging_config.every_steps != 0:
            return

        seconds_per_step = (time.time() - self.current_state.last_log
                            ) / self.config.logging_config.every_steps

        self.average_logger.update({"time_per_step": seconds_per_step})
        averaged_metrics = self.average_logger.average()

        # Write to tensorboard
        for k, v in averaged_metrics.items():
            self.tensorboard_writer.add_scalar(f"train/{k}", v,
                                               self.current_state.train_step)

        self._log_stats(averaged_metrics, prefix='train')
        self.current_state.last_log = time.time()

    def _log_stats(self, stats: Dict[str, float], prefix: str):
        msg_start = f'[{prefix}] Step {self.current_state.train_step} out of {self.config.optim_config.total_steps}' + ' | '
        dict_msg = ' | '.join(
            [f'{k.capitalize()} --> {v:.6f}'
             for k, v in stats.items()]) + ' | '

        msg = msg_start + dict_msg
        logger.info(msg)

    def _write_eval_results(self, prefix: str, eval_results: Dict[str, float]):
        eval_result_path = to_absolute_path(
            self.config.eval_config.eval_results_path)
        eval_dir, _ = os.path.split(eval_result_path)
        os.makedirs(eval_dir, exist_ok=True)

        with open(eval_result_path, 'a') as f:
            msg = f'[{prefix}] Step {self.current_state.train_step} out of {self.config.optim_config.total_steps}' + ' | '
            dict_msg = ' | '.join([
                f'{k.capitalize()} --> {v:.6f}'
                for k, v in eval_results.items()
            ]) + ' | '
            msg = msg + dict_msg
            f.write(msg + '\n')

    def _maybe_validate(self, **kwargs):
        if (self.current_state.train_step >
                self.config.optim_config.total_steps
                or self.current_state.train_step %
                self.config.eval_config.every_steps == 0):
            self.evaluate(
                **kwargs,
                prefix='validation',
            )

    def _maybe_save_checkpoint(self, accelerator: Accelerator):
        if (self.current_state.train_step >
                self.config.optim_config.total_steps
                or self.current_state.train_step %
                self.config.checkpoint.every_steps == 0):

            output_dir = f'checkpoint-{self.current_state.train_step}'
            if self.config.checkpoint.path is not None:
                output_dir = os.path.join(self.config.checkpoint.path,
                                          output_dir)
                output_dir = to_absolute_path(output_dir)

            accelerator.save_state(output_dir=output_dir)


def validate_config(config: TrainConfig):
    #TODO(hyeontae): Implement validation logic
    pass


def get_token_importance(
    tokenizer: PreTrainedTokenizer,
    representation: Representation,
) -> torch.FloatTensor:
    vocab_size = len(tokenizer)
    token_importance_map = torch.zeros(vocab_size, dtype=torch.float)
    tokens = [
        tokenizer.convert_ids_to_tokens(token_id)
        for token_id in range(vocab_size)
    ]
    token_importances = get_importance(
        tokens=tokens,
        representation=representation,
    )
    for token_id, token_importance in zip(range(vocab_size),
                                          token_importances):
        token_importance_map[token_id] = token_importance
    return token_importance_map


def update_temperature(loss_config: LossConfig, current_step: int):
    temp_scheduler_config = loss_config.temperature_scheduler_config
    if temp_scheduler_config is None:
        return

    decay_rate = temp_scheduler_config.decay_rate
    every_steps = temp_scheduler_config.every_steps
    min_temperature = temp_scheduler_config.min_temperature

    temperature = loss_config.temperature
    if current_step % every_steps == 0:
        if temp_scheduler_config.decay == TemperatureDecay.LINEAR.value:
            temperature -= decay_rate
        elif temp_scheduler_config.decay == TemperatureDecay.EXPONENTIAL.value:
            temperature *= math.exp(-decay_rate)
        else:
            raise ValueError(
                f"Invalid temperature decay: {temp_scheduler_config.decay}")

    if min_temperature is not None:
        temperature = max(temperature, min_temperature)

    loss_config.temperature = temperature


class TaskHelper:

    @staticmethod
    def set_task_metrics(test_task: TestTask) -> evaluate.Metric:
        task_metric_path_dict = {
            TestTask.MOL2TEXT.value: "metrics/translation_metrics",
            TestTask.TEXT2MOL.value: "metrics/save_only_metrics",
            TestTask.TEXT2FRAG.value: "metrics/save_only_metrics",
            TestTask.DTI.value: "metrics/dti_metrics",
            TestTask.PEER.value: "metrics/dti_metrics",
            TestTask.MOLNET.value: "metrics/dti_metrics",
        }
        if test_task not in task_metric_path_dict:
            raise ValueError(f"Invalid test task: {test_task}")

        metric = evaluate.load(
            to_absolute_path(task_metric_path_dict[test_task]))
        return metric

    @staticmethod
    def generate_results(test_task: TestTask, model: Model,
                         data_config: DataConfig,
                         batch: Dict[str, torch.Tensor]) -> GenerationResult:
        generation_result = GenerationResult(predictions=None, scores=None)

        #TODO(hyeontae): Remove hard-coded test tasks
        if test_task in [
                TestTask.DTI.value, TestTask.PEER.value, TestTask.MOLNET.value
        ]:
            results = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=data_config.max_target_len,
                generation_config=model.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generation_result.predictions = results.sequences
            generation_result.scores = results.scores
        else:
            generation_result.predictions = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=data_config.max_target_len,
            )

        return generation_result

    @staticmethod
    def parse(
        test_task: TestTask,
        representation: Representation,
        inputs: List[str],
        predictions: List[str],
        references: List[str],
        scores: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[str]]:

        # TODO(hyeontae): Remove hard-coded logic
        assert len(inputs) == len(predictions) == len(references)

        parsed_inputs, parsed_predictions, parsed_references \
            = None, None, None

        # Mol2Text, Only parse inputs
        if test_task == TestTask.MOL2TEXT.value:
            parsed_inputs = [
                input.split('- Input: ')[-1].split(' Output:')[0]
                for input in inputs
            ]
            parsed_predictions = predictions
            parsed_references = references

        # Text2Mol, Parse inputs, predictions, representations
        elif test_task == TestTask.TEXT2MOL.value:
            parsed_inputs = [
                input.split('- Input: ')[-1].split(' Output:')[0]
                for input in inputs
            ]

            parsed_predictions = []
            for prediction in predictions:
                parsed_prediction = representation.decode(prediction)
                parsed_predictions.append(parsed_prediction)

            # TODO(hyeontae): check references are SELFIES format
            selfies = Selfies()
            parsed_references = [
                selfies.decode(reference, verbose=True)
                for reference in references
            ]

        # Only parse predictions
        elif test_task in [
                TestTask.DTI.value, TestTask.PEER.value, TestTask.MOLNET.value
        ]:
            # No: 465, Yes: 2163
            parsed_predictions = [
                (scores[0][i][2163] /
                 (scores[0][i][2163] + scores[0][i][465])).item()
                for i in range(len(predictions))
            ]
            parsed_inputs = inputs
            parsed_references = references
        else:
            raise ValueError(f"Invalid test task: {test_task}")

        return parsed_inputs, parsed_predictions, parsed_references

    @staticmethod
    def set_additional_metrics(
        test_task: TestTask,
        predictions: List[str],
        references: List[str],
    ):
        additional_metrics = {}
        #TODO(hyeontae): Remove hard-coded logic
        if test_task == TestTask.TEXT2MOL.value:
            additional_metrics.update(
                get_text2mol_metrics(
                    predictions=predictions,
                    references=references,
                ))
        elif test_task == TestTask.MOL2TEXT.value:
            pass
        elif test_task == TestTask.TEXT2FRAG.value:
            pass
        elif test_task in [
                TestTask.DTI.value, TestTask.PEER.value, TestTask.MOLNET.value
        ]:
            pass
        else:
            raise ValueError(f"Invalid test task: {test_task}")

        return additional_metrics
