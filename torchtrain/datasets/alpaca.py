# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List

import torch
from torch.utils.data import DataLoader, IterableDataset

from torchtrain.datasets.tokenizer import TokenizerIf

from datasets import load_dataset


class AlpacaDataset(IterableDataset):
    """PyTorch Representation of the Alpaca Dataset from Hugging Face.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process

    Data input format:
    {
        "instruction": "Create a classification task by clustering the given list of items.",
        "input": "Apples, oranges, bananas, strawberries, pineapples",
        "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
        "text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a classification task by clustering the given list of items.\n\n### Input:\nApples, oranges, bananas, strawberries, pineapples\n\n### Response:\nClass 1: Apples,
        Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",  # noqa: B950
    }

    Example:
    >>> alpaca_ds = AlpacaDataset(tokenizer=tokenizer)
    >>> for batch in Dataloader(alpaca_ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(
        self,
        tokenizer: TokenizerIf,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        **kwargs
    ) -> None:
        self._data = load_dataset("tatsu-lab/alpaca", split="train")
        self._tokenizer = tokenizer
        self.data_iterator = iter(self._data)
        self.seq_len = seq_len
        self.world_size = world_size
        self.rank = rank
        self.response_tag = "\n\n### Response:\n"

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        all_tokens: List[int] = []

        for idx, sample in enumerate(self.data_iterator):
            # select samples to pack in a round-robin fashion
            # TODO: This is a temporary solution for small datasets like Alpaca.
            #       For larger datasets we need to use a more scalable approach.
            if idx % self.world_size != self.rank:
                continue
            sample_text = sample["text"]
            sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
            all_tokens.extend(sample_tokens)

            if len(all_tokens) >= max_buffer_token_len:
                x = torch.LongTensor(all_tokens[:max_buffer_token_len])
                # batched_x = x.reshape(self.batch_size, -1)
                # update tokens to the remaining tokens
                all_tokens = all_tokens[max_buffer_token_len:]
                input = x[:-1]
                label = x[1:]
                yield input, label


def build_alpaca_data_loader(
    tokenizer: TokenizerIf, batch_size: int, seq_len: int, world_size, rank
):
    alpaca_ds = AlpacaDataset(tokenizer, seq_len, world_size, rank)

    return DataLoader(alpaca_ds, batch_size=batch_size)
