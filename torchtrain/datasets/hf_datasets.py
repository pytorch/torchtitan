# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from torchtrain.datasets.tokenizer import TokenizerIf
from torchtrain.logging_utils import rank0_log

from datasets import load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node


class HuggingFaceDataset(IterableDataset):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset (Dataset): HuggingFace dataset downloaded from HuggingFace hub or loaded from disk.
        column (str): Column name of the dataset to iterate through.
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite: whether to loop infinitely over the dataset

    Example data input format (from the Alpaca dataset):
    {
        "instruction": "Create a classification task by clustering the given list of items.",
        "input": "Apples, oranges, bananas, strawberries, pineapples",
        "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
        "text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a classification task by clustering the given list of items.\n\n### Input:\nApples, oranges, bananas, strawberries, pineapples\n\n### Response:\nClass 1: Apples,
        Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",  # noqa: B950
    }

    Example:
    >>> alpaca_ds = HuggingFaceDataset.from_dataset_name(
            "tatsu-lab/alpaca", "text", tokenizer)
    >>> for batch in Dataloader(alpaca_ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(self, dataset, column, tokenizer, seq_len, world_size, rank, infinite):
        # TODO: This is a temporary solution for small datasets like Alpaca.
        #       For larger datasets we need to use a more scalable approach.
        self._data = split_dataset_by_node(dataset, rank, world_size)
        self._column = column
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        all_tokens: List[int] = []

        while True:
            for sample in iter(self._data):
                sample_text = sample[self._column]
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                all_tokens.extend(sample_tokens)

                while len(all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    all_tokens = all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label
            if not self.infinite:
                break

    @classmethod
    def from_dataset_name(
        cls,
        dataset_name: str,
        column: str,
        tokenizer: TokenizerIf,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        **kwargs,
    ):
        dataset = load_dataset(dataset_name, split="train")
        return cls(dataset, column, tokenizer, seq_len, world_size, rank, infinite)

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str,
        column: str,
        tokenizer: TokenizerIf,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        **kwargs,
    ):
        dataset = load_from_disk(dataset_path)
        return cls(dataset, column, tokenizer, seq_len, world_size, rank, infinite)


def build_alpaca_data_loader(
    tokenizer: TokenizerIf,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    dataset_path: Optional[str],
    infinite: bool = True,
):
    if dataset_path:
        rank0_log(f"Loading training data locally from {dataset_path}...")
        alpaca_ds = HuggingFaceDataset.from_dataset_path(
            dataset_path, "text", tokenizer, seq_len, world_size, rank, infinite
        )
    else:
        rank0_log(f"Downloading training data from HuggingFace...")
        alpaca_ds = HuggingFaceDataset.from_dataset_name(
            "tatsu-lab/alpaca", "text", tokenizer, seq_len, world_size, rank, infinite
        )

    return DataLoader(alpaca_ds, batch_size=batch_size)
