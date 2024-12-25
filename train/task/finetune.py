import logging
from typing import Tuple

import datasets
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import SpecialTokensMixin

from core.task import BaseTaskCls
from model.config import ModelConfig
from model.loader import ModelLoader
from model.representation import Representation
from train.config import DataConfig, TrainConfig
from train.train import Trainer, validate_config
from train.utils import DataCollatorForTextToMol
from utils import to_absolute_path

logger = logging.getLogger(__name__)


class App(BaseTaskCls):

    def __init__(
        self,
        train_config: TrainConfig,
        model_config: ModelConfig,
        data_config: DataConfig,
        **kwargs,
    ):
        super(App, self).__init__(**kwargs)
        self.train_config = train_config
        self.model_config = model_config
        self.data_config = data_config
        set_seed(self.train_config.seed)

    def run(self, **kwargs):
        validate_config(self.train_config)

        self.model_loader = ModelLoader(self.model_config)
        model = self.model_loader.load_model()
        representation = self.model_loader.load_representation()
        tokenizer = self.model_loader.load_tokenizer()

        self.trainer = Trainer(self.train_config, self.data_config)
        train_dataloader, test_dataloader, eval_dataloader = self.get_dataloader(
            tokenizer, representation)

        self.trainer.train(
            model=model,
            tokenizer=tokenizer,
            representation=representation,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            eval_dataloader=eval_dataloader,
        )

    def get_dataloader(
        self, tokenizer: SpecialTokensMixin, representation: Representation
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dataset = datasets.load_dataset(
            to_absolute_path(self.data_config.exec_file_path),
            data_dir=to_absolute_path(self.data_config.data_dir),
            task_dir=to_absolute_path(self.data_config.task_dir),
            max_num_instances_per_task=self.data_config.
            max_num_instances_per_task,
            max_num_instances_per_eval_task=self.data_config.
            max_num_instances_per_eval_task,
        )

        # TODO(hyeontae): check the logic. Now it is hard coded.
        train_data_collator = DataCollatorForTextToMol(
            tokenizer=tokenizer,
            representation=representation,
            padding="longest",
            max_source_length=self.data_config.max_seq_len,
            max_target_length=self.data_config.max_target_len,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
            token_importance_config=self.data_config.token_importance_config,
        )

        val_data_collator = DataCollatorForTextToMol(
            tokenizer=tokenizer,
            representation=representation,
            padding="longest",
            max_source_length=self.data_config.max_seq_len,
            max_target_length=self.data_config.max_target_len,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        batch_size = self.train_config.batch_size // self.train_config.grad_acc

        # train
        train_data_loader = DataLoader(
            dataset["train"],
            shuffle=True,
            collate_fn=train_data_collator,
            batch_size=batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # validation
        eval_data_loader = DataLoader(
            dataset["validation"],
            shuffle=False,
            collate_fn=val_data_collator,
            batch_size=batch_size * self.train_config.test_bsz_multi,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # test
        test_data_loader = DataLoader(
            dataset["test"],
            shuffle=False,
            collate_fn=val_data_collator,
            batch_size=batch_size * self.train_config.test_bsz_multi,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_data_loader, test_data_loader, eval_data_loader
