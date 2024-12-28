import string
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
import torch
from datasets import IterableDataset
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.data.data_collator import *

from model.representation import Representation
from train.config import TokenImportanceConfig
from train.token_importance import get_token_importance

IMPORTANCE_PAD_VALUE = -1.0
SHUFFLE_BUFFER_SIZE = 10000


@dataclass
class DataCollatorForText2Mol:
    tokenizer: PreTrainedTokenizerBase
    representation: Representation
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    token_importance_config: TokenImportanceConfig = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "

            definition = ""
            if isinstance(instance["Definition"], list):
                definition = "Definition: " + instance["Definition"][0].strip()
            else:
                definition = "Definition: " + instance["Definition"].strip()
            if not definition[-1] in string.punctuation:
                definition += "."
            definition += "\n\n"

            source = definition + task_input
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(
                    self.tokenizer.decode(
                        tokenized_source[:self.max_source_length],
                        skip_special_tokens=True,
                    ))

        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        assert all(ex["Instance"].get("output") != None for ex in batch), \
            "output is required for data in TextToMol task."

        labels = [ex["Instance"]["output"][0] for ex in batch]
        token_importances = None
        if "importance" in batch[0]["Instance"]:
            token_importances = [
                ex["Instance"]["importance"][0] for ex in batch
            ]

        encoded_labels = self.tokenizer(
            labels,
            max_length=self.max_target_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        label_mask = encoded_labels["attention_mask"].bool()
        model_inputs["labels"] = encoded_labels["input_ids"].masked_fill(
            ~label_mask, self.label_pad_token_id)

        if self.token_importance_config is not None:
            tokenized_labels = [
                self.tokenizer.convert_ids_to_tokens(label)
                for label in encoded_labels["input_ids"]
            ]
            token_importanaces = get_token_importance(
                config=self.token_importance_config,
                tokenized_labels=tokenized_labels,
                tokenizer=self.tokenizer,
                representation=self.representation,
                predefined_importances=token_importances,
            )
            token_importanaces = torch.Tensor(token_importanaces)
            model_inputs["token_importances"] = token_importanaces.masked_fill(
                ~label_mask, IMPORTANCE_PAD_VALUE)

        return model_inputs


@dataclass
class DataCollatorForT5MLM:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
    """

    tokenizer: PreTrainedTokenizer
    representation: Representation
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    token_importance_config: TokenImportanceConfig = None

    def __call__(
        self, examples: List[Tuple[Dict[str, Any],
                                   Dict[str, Any]]]) -> BatchEncoding:
        text_examples, mol_examples = zip(*examples)

        text_input_ids = [example["input_ids"] for example in text_examples]
        mol_input_ids = [example["input_ids"] for example in mol_examples]

        all_ids = text_input_ids + mol_input_ids
        first_length = len(all_ids[0])
        assert all(
            len(ids) == first_length
            for ids in all_ids), "All input_ids must have the same length"

        batch = BatchEncoding({
            "input_ids": np.array(all_ids),
        })

        input_ids = batch["input_ids"]
        batch_size, expanded_input_length = input_ids.shape

        mask_indices = np.asarray([
            self.random_spans_noise_mask(expanded_input_length)
            for _ in range(batch_size)
        ])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(
            mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids,
                                                   input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        # Token importance
        if self.token_importance_config is not None:
            token_importances = self.get_token_importance(
                batch_size=batch_size,
                expanded_input_length=expanded_input_length,
                mol_examples=mol_examples,
            )
            batch["token_importances"] = self.filter_token_importances(
                token_importances, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}.")

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}.")

        if self.token_importance_config is not None and batch[
                "token_importances"].shape[-1] != self.target_length:
            raise ValueError(
                f"`token_importances` are incorrectly preprocessed. `token_importances` length is {batch['token_importances'].shape[-1]}, but should be"
                f" {self.target_length}.")

        batch = BatchEncoding(
            {k: torch.from_numpy(v)
             for k, v in batch.items()})
        return batch

    def get_token_importance(
        self,
        batch_size: int,
        expanded_input_length: int,
        mol_examples: List[Dict[str, Any]],
    ) -> torch.Tensor:
        text_batch_size = batch_size // 2
        text_token_importances = torch.ones(
            (text_batch_size, expanded_input_length))

        mol_tokens = [
            self.tokenizer.convert_ids_to_tokens(example["input_ids"])
            for example in mol_examples
        ]
        mol_token_importances = get_token_importance(
            config=self.token_importance_config,
            tokenized_labels=mol_tokens,
            tokenizer=self.tokenizer,
            representation=self.representation,
        )
        mol_token_importances = torch.Tensor(mol_token_importances)
        token_importances = torch.cat(
            [text_token_importances, mol_token_importances], dim=0)

        return token_importances

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1,
                                               axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0,
                                np.cumsum(start_indices, axis=-1),
                                start_indices)
        sentinel_ids = np.where(sentinel_ids != 0,
                                (self.tokenizer.vocab_size - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1),
                        self.tokenizer.eos_token_id,
                        dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def filter_token_importances(self, token_importances, sentinel_ids):
        """
        Puts sentinel mask on `token_importances` and fuse consecutive mask tokens importances into a 1 by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = token_importances.shape[0]
        mask_importance = self.token_importance_config.special_token_importance
        token_importances_full = np.where(sentinel_ids != 0, sentinel_ids,
                                          token_importances)
        token_importances_full = np.where(sentinel_ids > 0, mask_importance,
                                          token_importances_full)

        # token_importances tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        token_importances = token_importances_full[
            token_importances_full >= 0].reshape((batch_size, -1))
        token_importances = np.concatenate(
            [
                token_importances,
                np.full((batch_size, 1), mask_importance, dtype=np.float32),
            ],
            axis=-1,
        )
        return token_importances

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens,
                                                  num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                     num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length, ), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


def build_mlm_dataset(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizer,
    input_length: int,
    shuffle: bool = False,
) -> IterableDataset:
    dataset = dataset.map(
        _mlm_tokenize_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "input_length": input_length,
        },
        remove_columns=["text"],
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    return dataset


def _mlm_tokenize_function(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    input_length: int,
):
    output = tokenizer(
        text=examples["text"],
        return_attention_mask=False,
    )
    input_ids = output["input_ids"]
    concatenated_ids = np.concatenate(input_ids)

    total_length = concatenated_ids.shape[0]
    total_length = (total_length // input_length) * input_length

    concatenated_ids = concatenated_ids[:total_length].reshape(
        -1, input_length)
    return {"input_ids": concatenated_ids}


def get_input_and_target_lengths(
    input_length: int,
    noise_density: float,
    mean_noise_span_length: float,
) -> Tuple[int, int]:
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        input_length: desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: length in tokens of encoded targets sequence
    """

    def _tokens_length_to_input_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = input_length

    while _tokens_length_to_input_length_targets_length(tokens_length +
                                                        1)[0] <= input_length:
        tokens_length += 1

    input_length, targets_length = _tokens_length_to_input_length_targets_length(
        tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > input_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


class MixedDataset(IterableDataset):

    def __init__(
        self,
        text_dataset: IterableDataset,
        mol_dataset: IterableDataset,
    ):
        self.text_dataset = text_dataset
        self.mol_dataset = mol_dataset

    def _reset_iterator(
        self,
        iterator: Iterator,
        dataset: IterableDataset,
    ):
        """
        Helper function to reset the iterator if StopIteration occurs.
        """
        try:
            return next(iterator)
        except StopIteration:
            return next(iter(dataset))

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        text_iter = iter(self.text_dataset)
        molecule_iter = iter(self.mol_dataset)

        while True:
            # Fetch the next batch or reset the iterator
            text_batch = self._reset_iterator(text_iter, self.text_dataset)
            molecule_batch = self._reset_iterator(molecule_iter,
                                                  self.mol_dataset)

            # Return the combined batch
            yield text_batch, molecule_batch
