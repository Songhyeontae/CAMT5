from typing import Tuple

import datasets
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import SpecialTokensMixin

import train.data as data_utils
from core.task import BaseTaskCls
from model.config import ModelConfig, RepresentationType
from model.loader import ModelLoader
from model.representation import Representation
from train.config import DataConfig, TrainConfig
from train.data import DataCollatorForUnimptT5, MixedDatasetWithT2M
from train.train import Trainer, validate_config
from utils import to_absolute_path


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

        self.trainer = Trainer(self.train_config)
        train_dataloader, test_dataloader = self.get_dataloader(
            tokenizer, representation)

        self.trainer.train(
            model=model,
            tokenizer=tokenizer,
            representation=representation,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        )

    def get_dataloader(
            self, tokenizer: SpecialTokensMixin,
            representation: Representation) -> Tuple[DataLoader, DataLoader]:
        pretrain_data_config = self.data_config.continual_pretrain_data_config

        c4_dataset = self.get_text_dataset()
        mol_dataset = self.get_mol_dataset(representation)
        text2mol_dataset = self.get_text2mol_dataset(
            pretrain_data_config.t2m_data_path, representation)

        extended_input_length, mlm_target_length = data_utils.get_input_and_target_lengths(
            input_length=pretrain_data_config.input_length,
            noise_density=pretrain_data_config.mlm_probability,
            mean_noise_span_length=pretrain_data_config.mean_noise_span_length,
        )

        # Set the max_target_len
        self.train_config.eval_config.max_target_len = pretrain_data_config.input_length

        text_mlm_dataset = data_utils.build_mlm_dataset(
            dataset=c4_dataset,
            tokenizer=tokenizer,
            input_length=extended_input_length,
            shuffle=True,
        )
        mol_mlm_dataset = data_utils.build_mlm_dataset(
            dataset=mol_dataset,
            tokenizer=tokenizer,
            input_length=extended_input_length,
            shuffle=True,
        )

        text2mol_dataset = data_utils.build_text2mol_dataset(
            dataset=text2mol_dataset,
            tokenizer=tokenizer,
            input_length=pretrain_data_config.input_length,
            shuffle=True,
        )

        train_mixed_mlm_dataset = MixedDatasetWithT2M(
            text_mlm_dataset["train"], mol_mlm_dataset["train"],
            text2mol_dataset["train"])
        test_mixed_mlm_dataset = MixedDatasetWithT2M(
            text_mlm_dataset["validation"], mol_mlm_dataset["validation"],
            text2mol_dataset["train"])

        batch_size = self.train_config.batch_size // self.train_config.grad_acc

        # compute ratio of datas
        t2m_data_ratio = pretrain_data_config.t2m_data_ratio
        text_data_ratio = (1 - t2m_data_ratio) / 2
        mol_data_ratio = (1 - t2m_data_ratio) / 2

        data_collator = DataCollatorForUnimptT5(
            tokenizer=tokenizer,
            representation=representation,
            noise_density=pretrain_data_config.mlm_probability,
            mean_noise_span_length=pretrain_data_config.mean_noise_span_length,
            input_length=pretrain_data_config.input_length,
            target_length=mlm_target_length,
            pad_token_id=tokenizer.pad_token_id,
            text_data_ratio=text_data_ratio,
            mol_data_ratio=mol_data_ratio,
            t2m_data_ratio=t2m_data_ratio,
            token_importance_config=self.data_config.token_importance_config,
        )

        train_dataloader = DataLoader(
            train_mixed_mlm_dataset,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=5,
        )
        test_dataloader = DataLoader(
            test_mixed_mlm_dataset,
            collate_fn=data_collator,
            batch_size=batch_size * self.train_config.test_bsz_multi,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=5,
        )

        return train_dataloader, test_dataloader

    def get_text_dataset(self) -> datasets.IterableDataset:
        c4_dataset = datasets.load_dataset("c4", "en", streaming=True)
        c4_dataset = c4_dataset.remove_columns(["timestamp", "url"])
        return c4_dataset

    def get_mol_dataset(
            self, representation: Representation) -> datasets.IterableDataset:
        zinc_dataset = datasets.load_dataset('zpn/zinc20', streaming=True)
        if self.model_config.representation_type == RepresentationType.SMILES.value:
            zinc_dataset = zinc_dataset.remove_columns(["id", "selfies"])
            zinc_dataset = zinc_dataset.rename_column("smiles", "text")
            zinc_dataset = zinc_dataset.map(
                lambda example: {
                    "text":
                    _molecule_process(representation.encode(example["text"]))
                }, )

        elif self.model_config.representation_type == RepresentationType.SELFIES.value:
            zinc_dataset = zinc_dataset.remove_columns(["id", "smiles"])
            zinc_dataset = zinc_dataset.rename_column("selfies", "text")
            zinc_dataset = zinc_dataset.map(
                lambda example: {"text": _molecule_process(example["text"])}, )
        elif self.model_config.representation_type == RepresentationType.FRAG.value:
            zinc_dataset = zinc_dataset.remove_columns(["id", "selfies"])
            zinc_dataset = zinc_dataset.rename_column("smiles", "text")
            zinc_dataset = zinc_dataset.map(
                lambda example: {
                    "text":
                    _molecule_process(representation.encode(example["text"]))
                }, )

        return zinc_dataset

    def get_text2mol_dataset(
        self,
        data_path: str,
        representation: Representation,
    ) -> datasets.Dataset:
        data_path = to_absolute_path(data_path)
        dataset = datasets.load_dataset("csv",
                                        data_files=[data_path],
                                        delimiter="\t")

        if self.model_config.representation_type == RepresentationType.SMILES.value:
            dataset = dataset.map(
                lambda example: {
                    "desc": example["desc"],
                    "seq": _molecule_process(example["smiles"]),
                }, )

        elif self.model_config.representation_type == RepresentationType.SELFIES.value:
            dataset = dataset.map(
                lambda example: {
                    "desc": example["desc"],
                    "seq": _molecule_process(example["selfies"]),
                }, )

        elif self.model_config.representation_type == RepresentationType.FRAG.value:
            dataset = dataset.map(
                lambda example: {
                    "desc":
                    example["desc"],
                    "seq":
                    _molecule_process(representation.encode(example["smiles"])
                                      ),
                }, )
        else:
            raise ValueError(
                f"Invalid representation type: {self.model_config.representation_type}"
            )
        dataset.remove_columns(["smiles", "selfies"])

        return dataset


def _molecule_process(sequence: str) -> str:
    return "<bom>" + sequence + "<eom>"
