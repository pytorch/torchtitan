# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List

import torch
from torch.utils.data import DataLoader, IterableDataset

from torchtrain.datasets.tokenizer import TokenizerIf

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

_supported_datasets = {
    "alpaca": "tatsu-lab/alpaca",
    "minipile": "JeanKaddour/minipile",
}


class HuggingFaceDataset(IterableDataset):
    """PyTorch Representation of a Dataset from Hugging Face.

    We currently support two datasets:
    minipile (1M training entries)
    alpaca (57K training entries)

    >> MiniPile <<:
    MiniPile dataset is detailed in the following paper:
    https://arxiv.org/abs/2304.08442

    Args:
        dataset_name (str): name of the dataset to load
        tokenizer (TokenizerIf): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    Data input format (minipile):
    {
        "text": "Open-end spinning devices with such rotor bearing arrangements are known in
                various different embodiments, and have been extensively described,
                for example in German Patent Publications"
    }

    >> Alpaca <<:
    Data input format (alpaca):
    {
        "instruction": "Create a classification task by clustering the given list of items.",
        "input": "Apples, oranges, bananas, strawberries, pineapples",
        "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
        "text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a classification task by clustering the given list of items.\n\n### Input:\nApples, oranges, bananas, strawberries, pineapples\n\n### Response:\nClass 1: Apples,
        Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",  # noqa: B950
    }

    Example:
    >>> alpaca_ds = HuggingFaceDataset(tokenizer=tokenizer)
    >>> for batch in Dataloader(alpaca_ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: TokenizerIf,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        # TODO: This is a temporary solution for small datasets like Alpaca.
        #       For larger datasets we need to use a more scalable approach.
        # Setting `streaming=True` works for large dataset, but the speed is slow.
        if dataset_name not in _supported_datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not supported. Supported datasets are: {_supported_datasets.keys()}"
            )

        ds = load_dataset(_supported_datasets[dataset_name], split="train")
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        all_tokens: List[int] = []

        while True:
            for sample in iter(self._data):
                sample_text = sample["text"]
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


def build_hf_data_loader(
    dataset_name: str,
    tokenizer: TokenizerIf,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True,
):
    hf_ds = HuggingFaceDataset(dataset_name, tokenizer, seq_len, world_size, rank, infinite)

    return DataLoader(hf_ds, batch_size=batch_size)
