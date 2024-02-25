# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List

import torch
from torch.utils.data import DataLoader, IterableDataset

from torchtrain.datasets.tokenizer import TokenizerIf

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


class MiniPileDataset(IterableDataset):
    """PyTorch Representation of the MiniPile Dataset from Hugging Face.
    MiniPile dataset is detailed in the following paper:
    https://arxiv.org/abs/2304.08442

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process

    Data input format:
    {
        "text": "Open-end spinning devices with such rotor bearing arrangements are known in
                various different embodiments, and have been extensively described,
                for example in German Patent Publications"
    }

    Example:
    >>> minipile_ds = MiniPileDataset(tokenizer=tokenizer)
    >>> for batch in Dataloader(minipile_ds, batch_size=8):
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
        # TODO: This is a temporary solution for small datasets like Alpaca.
        #       For larger datasets we need to use a more scalable approach.
        # Setting `streaming=True` works for large dataset, but the speed is slow.
        ds = load_dataset("JeanKaddour/minipile", split="train")
        self.data_iterator = iter(split_dataset_by_node(ds, rank, world_size))
        self._tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        all_tokens: List[int] = []

        for sample in self.data_iterator:
            sample_text = sample["text"]
            sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
            all_tokens.extend(sample_tokens)

            while len(all_tokens) >= max_buffer_token_len:
                x = torch.LongTensor(all_tokens[:max_buffer_token_len])
                # batched_x = x.reshape(self.batch_size, -1)
                # update tokens to the remaining tokens
                all_tokens = all_tokens[max_buffer_token_len:]
                input = x[:-1]
                label = x[1:]
                yield input, label


def build_minipile_data_loader(
    tokenizer: TokenizerIf, batch_size: int, seq_len: int, world_size, rank
):
    minipile_ds = MiniPileDataset(tokenizer, seq_len, world_size, rank)

    return DataLoader(minipile_ds, batch_size=batch_size)
