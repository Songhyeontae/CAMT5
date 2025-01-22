import logging
from typing import Union

import torch
from transformers import (AutoConfig, AutoTokenizer, OPTForCausalLM,
                          PretrainedConfig, PreTrainedTokenizer,
                          T5ForConditionalGeneration)

from model.config import ModelConfig, TokenizerConfig
from model.representation import Representation, get_representation
from utils import to_absolute_path

logger = logging.getLogger(__name__)

Model = Union[T5ForConditionalGeneration, OPTForCausalLM]


class ModelLoader:

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def _get_pretrained_model_config(self) -> PretrainedConfig:
        config = AutoConfig.from_pretrained(self.model_config.name)
        # Reset the dropout_rate
        config.dropout_rate = self.model_config.dropout
        return config

    def _load_t5_model(self) -> Model:
        pt_model_config = self._get_pretrained_model_config()
        if not self.model_config.load_model.from_pretrained:
            model = T5ForConditionalGeneration(pt_model_config)
        elif (self.model_config.load_model.from_pretrained
              and "facebook" in self.model_config.name):
            model = OPTForCausalLM.from_pretrained(self.model_config.name)
        elif self.model_config.load_model.from_pretrained:
            model = T5ForConditionalGeneration.from_pretrained(
                self.model_config.name, config=pt_model_config)
        else:
            raise ValueError("Invalid model configuration")
        return model

    def _add_special_tokens(self, tokenizer_config: TokenizerConfig,
                            tokenizer: PreTrainedTokenizer):
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        tokenizer.model_max_length = int(1e9)
        AMINO_ACIDS = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
        prefixed_amino_acids = [f"<p>{aa}" for aa in AMINO_ACIDS]
        tokenizer.add_tokens(prefixed_amino_acids)

        for additional_tokens_path in tokenizer_config.additional_tokens_paths:
            additional_tokens_path = to_absolute_path(additional_tokens_path)
            vocabs = [line.strip() for line in open(additional_tokens_path)]
            tokenizer.add_tokens(vocabs)

        special_tokens_dict = {
            "additional_special_tokens": [
                "<bom>",
                "<eom>",
                "<bop>",
                "<eop>",
                "MOLECULE NAME",
                "DESCRIPTION",
                "PROTEIN NAME",
                "FUNCTION",
                "SUBCELLULAR LOCATION",
                "PROTEIN FAMILIES",
            ]
        }
        tokenizer.add_special_tokens(special_tokens_dict,
                                     replace_additional_special_tokens=False)

        return tokenizer

    def load_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer_config = self.model_config.tokenizer_config
        if tokenizer_config.model_max_length is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.name,
                model_max_length=tokenizer_config.model_max_length)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.name)
        if tokenizer_config.add_special_tokens:
            tokenizer = self._add_special_tokens(tokenizer_config, tokenizer)

        return tokenizer

    def load_model(self) -> Model:
        tokenizer = self.load_tokenizer()
        model = self._load_t5_model()
        model.resize_token_embeddings(len(tokenizer))

        # check whether checkpoint exists
        if self.model_config.checkpoint_path:
            checkpoint_path = self.model_config.checkpoint_path
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            model.load_state_dict(torch.load(checkpoint_path,
                                             map_location=map_location),
                                  strict=True)
            torch.cuda.empty_cache()

            logger.info(f"Loaded model from checkpoint: {checkpoint_path}")

        return model

    def load_representation(self) -> Representation:

        return get_representation(self.model_config.representation_type)
