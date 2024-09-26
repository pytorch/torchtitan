# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from typing import Any, Dict, List, Optional
from pathlib import Path
import glob
import os

import numpy as np

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

try:
    from torchdata.stateful_dataloader import StatefulDataLoader
except ImportError as e:
    raise ImportError(
        "Please install the latest torchdata nightly to use StatefulDataloader via:"
        "pip3 install --pre torchdata --index-url https://download.pytorch.org/whl/nightly"
    ) from e

from torchtitan.tokenizers.tokenizer import Tokenizer
from torchtitan.logging import logger
from torchtitan.utils.dataset_utils import chemlactica_style_data_processing,create_fresh_file_store

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

# map from dataset name to a local directory, or
# a dataset repository on the HF hub
_supported_datasets = {
    "c4_test": "test/assets/c4_test",
    "c4": "allenai/c4",
    "chemlactica_train_mini": "test/assets/chemlactica_train_mini",
    "chemlactica_train": "/nfs/dgx/raid/chem/data/rdkit_computed_rel+form/train_rdkit_computed_rel+form"
}

_supported_data_processing_styles = {
    "chemlactica_style": chemlactica_style_data_processing
}


class HuggingFaceDataset(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]):
            Path to the dataset in the file system. If provided, data will be loaded
            from this path instead of downloaded.
        data_processing_style (str): name of the data process style    
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    We currently support the c4 dataset, and a subset of it for testing purposes:
    c4_test (2K training entries)
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
    >>> ds = HuggingFaceDataset(dataset_name="c4", dataset_path=None, data_processing_style="chemlactica_style", tokenizer=tokenizer)
    >>> for batch in Dataloader(ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        data_processing_style: str,
        tokenizer: Tokenizer,
        representation_type: str = "SMILES",
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        special_mode = None,
        store = None,
    ) -> None:
        # allow user to pass in a (local or HF hub) path to use unsupported datasets
        if dataset_name not in _supported_datasets:
            if dataset_path:
                logger.warning(
                    f"Dataset {dataset_name} is not tested or verfied. "
                    f"Recommended datasets are: {list(_supported_datasets.keys())}"
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} is not supported. "
                    f"Supported datasets are: {list(_supported_datasets.keys())}"
                )

        if not dataset_path:
            dataset_path = _supported_datasets[dataset_name]
        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")

        if dataset_name == "c4":
            # c4 is huge, and requires both streaming and language selection
            # (we default to en)
            ds = load_dataset(dataset_path, name="en", split="train", streaming=True)
        elif dataset_name == "c4_test":
            ds = load_dataset(dataset_path, split="train")
        else:
            dataset_files = glob.glob(os.path.join(dataset_path, "*.jsonl"))
            ds = load_dataset("text", data_files=dataset_files, split="train", streaming=True)
        try:
            data_processing_fn = _supported_data_processing_styles[data_processing_style]
        except KeyError as e:
            raise ValueError(f"Unsupported data processing style: {data_processing_style}")
        # data_processing_fn = lambda x, e: str(x)

        # TODO: support shuffling and checkpointing
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self.data_processing_fn = data_processing_fn
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.rank = rank
        self.world_size = world_size
        self.representation_type = representation_type

        # for non sync communication between ranks
        if not self.infinite and store:
            self.store = store
        else:
            self.store = None
    
        # variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []

        # random number generator
        self.rng = np.random.default_rng()

        # debugging dataloader yielding
        self.special_mode = str(special_mode)

    def _some_rank_finished(self) -> bool:
        if not self.infinite and self.store.num_keys() > 1: # one key used for coordination, more than one means one of the ranks exhausted data
            return True
        else:
            return False

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            if self.special_mode == "yield_tensor":
                logger.info("yielding tensor")
                yield random_tensor, random_tensor
                random_tensor = torch.randint(low=1, high=2, size=(self.seq_len,))
                continue

            for sample_json in self._get_data_iter():
                if self._some_rank_finished():
                    break
                sample_text = self.data_processing_fn(sample_json, self.rng, self.representation_type)
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                self._all_tokens.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label

            if not self.infinite:
                self.store.set(str(self.rank),"Done")
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                self.store.wait([str(k) for k in range(self.world_size)]) # making sure all ranks get to this point
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        # Skip samples
        if isinstance(self._data, IterableDataset):
            it = iter(self._data)
            # Naively iterate through the samples as skip may not be supported
            for _ in range(self._sample_idx):
                next(it)
            return it

        # As skipping to the end throws an error in case of map-style dataset, return an empty iterator
        if self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """
    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int, pin_memory: bool, num_workers: int):
        super().__init__(hf_ds, batch_size, num_workers=num_workers)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid, don't log a warning
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}"
            )
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    data_processing_style: str,
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    representation_type,
    infinite: bool = True,
    pin_memory: bool = False,
    num_workers: int = 2,
    special_mode = None,
    context = "train",
):
    if not infinite:
        store_identifier = f"rankstore_{context}_{dataset_name}"
        data_completion_store = create_fresh_file_store(store_identifier,world_size)
    else:
        data_completion_store = None

    hf_ds = HuggingFaceDataset(
        dataset_name, dataset_path, data_processing_style, tokenizer, representation_type, seq_len, world_size, rank, infinite, special_mode,store = data_completion_store
    )

    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
