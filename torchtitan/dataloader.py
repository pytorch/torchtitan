# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.datasets.tokenizer import Tokenizer


@dataclass
class BaseDataLoader(Stateful, ABC):
    """Base class for all dataloaders.

    This is used to enforce that all dataloaders have the methods defined in ``Stateful``,
    ``state_dict()`` and ``load_state_dict()``.
    """

    tokenizer: Tokenizer
    dp_rank: int
    dp_world_size: int
    batch_size: int

    @abstractmethod
    def __iter__(self):
        ...


class DPDataLoader(StatefulDataLoader, BaseDataLoader):
    """Dataloader that is aware of distributed data parallelism.

    This dataloader is used to load data in a distributed data parallel fashion. It also
    utilizes ``torchdata.stateful_dataloader.StatefulDataLoader`` to implement the necessary
    methods such as ``__iter__``.

    Args:
        dataset (IterableDataset): The dataset to iterate over.
        tokenizer (Tokenizer): The tokenizer to use to tokenize the dataset.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        batch_size: The batch size to use for each iteration.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        tokenizer: Tokenizer,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
    ):
        BaseDataLoader.__init__(
            self,
            tokenizer=tokenizer,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=batch_size,
        )
        StatefulDataLoader.__init__(self, dataset, batch_size)
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions.
        return {
            # We don't have to use pickle as DCP will serialize the state_dict. However,
            # we have to keep this for backward compatibility.
            self._rank_id: pickle.dumps(StatefulDataLoader.state_dict(self)),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # State being empty is valid.
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                "expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        # We don't have to use pickle as DCP will serialize the state_dict. However, we have to
        # keep this for backward compatibility.
        StatefulDataLoader.load_state_dict(
            self, pickle.loads(state_dict[self._rank_id])
        )


class DataLoaderBuilder(Protocol):
    """This is a protocol to annoate ``build_dataloader_fn``.

    While mypy.extensions provides Arg to annotate the name, it requires another dependency on
    mypy-extensions.  Mypy also supports this annonation and it is easier to read.
    """

    def __call__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer_path: str,
        batch_size: int,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
    ) -> BaseDataLoader:
        """Function call

        Args:
            dataset_name (str): Name of the dataset to iterate over.
            dataset_path (Optional[str]): Path to the dataset to load.
            tokenizer_path (str): Path to the tokenizer to use.
            batch_size (int): The batch size to use for each iteration.
            seq_len (int): Sequence length for each batch.
            dp_rank (int): Data parallelism rank for this dataloader.
            dp_world_size (int): The world size of the data parallelism.

        Returns:
            BaseDataLoader: The dataloader.
        """
        ...
