# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch

from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader, DistributedSampler

from torchtrain.datasets.tokenizer import TokenizerIf


class AlpacaDataset(IterableDataset):
    """PyTorch Representation of the Alpaca Dataset from Hugging Face.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length

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

    def __init__(self,
        tokenizer: TokenizerIf,
        seq_len: int = 2048,
        **kwargs
    ) -> None:
        self._data = load_dataset("tatsu-lab/alpaca", split="train")
        self._tokenizer = tokenizer
        self.data_iterator = iter(self._data)
        self.seq_len = seq_len
        self.response_tag = "\n\n### Response:\n"

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        max_buffer_token_len = (1 + self.seq_len)
        all_tokens: List[int] = []

        for sample in self.data_iterator:
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
    tokenizer: TokenizerIf,
    batch_size: int,
    seq_len: int,
    world_size,
    rank
):
    alpaca_ds = AlpacaDataset(tokenizer=tokenizer, seq_len=seq_len)
    # TOOD: sampler can't work with iterable dataset, figure out a way
    # to sample in a distributed manner
    # dist_sampler = DistributedSampler(
    #     alpaca_ds,
    #     world_size,
    #     rank,
    #     shuffle=True,
    # )

    return DataLoader(alpaca_ds, batch_size=batch_size)
