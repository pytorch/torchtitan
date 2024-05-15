# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging_utils import logger

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

# map from dataset name to a local directory, or
# a dataset repository on the HF hub
_supported_datasets = {
    "c4_mini": "torchtitan/datasets/c4_mini",
    "c4": "allenai/c4",
}


class HuggingFaceDataset(IterableDataset):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]):
            Path to the dataset in the file system. If provided, data will be loaded
            from this path instead of downloaded.
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    We currently support the c4 dataset and a subset of it:
    c4_mini (45K training entries)
    c4 (177M training entries - this dataset is streamed due to the size)

    >> c4 (EN) <<:
    c4 cleaned, English version
    Data input format (c4):
    {
    'url': 'https://klyq.com/beginners-bbq-class-taking-place-in-missoula/',
    'text': 'Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at ...',
    'timestamp': '2019-04-25T12:57:54Z'
    }

    Example use (c4):
    >>> ds = HuggingFaceDataset(dataset_name="c4", dataset_path=None, tokenizer=tokenizer)
    >>> for batch in Dataloader(ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        # allow user to pass in a (local or HF hub) path to use unsupported datasets
        if dataset_name not in _supported_datasets:
            if dataset_path:
                logger.warning(
                    f"Dataset {dataset_name} is not tested or verfied. "
                    f"Recommended datasets are: {list(_supported_datasets.keys())}."
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} is not supported. "
                    f"Supported datasets are: {list(_supported_datasets.keys())}."
                )

        if not dataset_path:
            dataset_path = _supported_datasets[dataset_name]
        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")

        if dataset_name == "c4":
            # c4 is huge, and requires both streaming and language selection
            # (we default to en)
            ds = load_dataset(dataset_path, name="en", split="train", streaming=True)
        else:
            ds = load_dataset(dataset_path, split="train")

        # TODO: support shuffling and checkpointing
        self.dataset_name = dataset_name
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
                logger.warning(f"Dataset {self.dataset_name} has run out of data.")
                break
            else:
                logger.warning(
                    f"Dataset {self.dataset_name} is being re-looped. "
                    "Loss related metrics might be misleading."
                )


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True,
):
    hf_ds = HuggingFaceDataset(
        dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, infinite
    )

    return DataLoader(hf_ds, batch_size=batch_size)
