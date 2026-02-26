# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import inspect
import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger


# NOTE: This class deliberately inherits from `Exception` and not `StopIteration`.
# According to PEP 479, raising a `StopIteration` or its subclass from within a
# generator will wrap it in a `RuntimeError`. Since this exception is designed
# to be raised from a generator-based dataloader and caught by the training loop,
# inheriting from `StopIteration` would make it uncatchable and would crash the
# program.
# See: https://peps.python.org/pep-0479/
class DataloaderExhaustedError(Exception):
    """An exception that indicates dataloader exhaustion."""

    pass


class BaseDataLoader(Stateful, ABC, Configurable):
    """Base class for all dataloaders.

    This is used to enforce that all dataloaders have the methods defined in ``Stateful``,
    ``state_dict()`` and ``load_state_dict()``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dataset: str = ""
        dataset_path: str | None = None

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        ...


class ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Dataloader that is aware of distributed data parallelism.

    This dataloader is used to load data in a distributed data parallel fashion. It also
    utilizes ``torchdata.stateful_dataloader.StatefulDataLoader`` to implement the necessary
    methods such as ``__iter__``.

    Args:
        dataset (IterableDataset): The dataset to iterate over.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        **kwargs: Additional keyword arguments passed to StatefulDataLoader (e.g.,
            batch_size, collate_fn, num_workers, persistent_workers, prefetch_factor,
            pin_memory).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        num_workers: int = 0
        """Number of worker processes for data loading."""

        persistent_workers: bool = False
        """Keep workers alive between epochs. Only valid when num_workers > 0."""

        pin_memory: bool = False
        """Copy tensors to CUDA pinned memory before returning them."""

        prefetch_factor: int | None = None
        """
        Number of batches loaded in advance by each worker. Only valid when num_workers > 0.
        Default is 2 when num_workers > 0, otherwise None.
        """

    dp_rank: int
    dp_world_size: int

    def __init__(
        self,
        dataset: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        **kwargs,
    ):
        self._validate_kwargs(kwargs)

        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

        super().__init__(dataset, **kwargs)

    @staticmethod
    def _validate_kwargs(kwargs: dict[str, Any]) -> None:
        """Validate and sanitize kwargs passed to the dataloader.

        Args:
            kwargs: Dictionary of keyword arguments to validate. This dict is
                modified in-place to remove invalid combinations.

        Raises:
            ValueError: If 'dataset' is in kwargs or if any invalid kwargs are passed.
        """
        if "dataset" in kwargs:
            raise ValueError(
                "'dataset' should not be passed in kwargs; "
                "it must be provided as the first positional argument."
            )

        sig = inspect.signature(StatefulDataLoader.__init__)
        valid_kwargs = frozenset(
            name for name in sig.parameters.keys() if name not in ("self", "dataset")
        )
        invalid_kwargs = set(kwargs.keys()) - valid_kwargs
        if invalid_kwargs:
            raise ValueError(
                f"Invalid dataloader kwargs: {invalid_kwargs}. "
                f"Valid kwargs are: {sorted(valid_kwargs)}"
            )

        # persistent_workers and prefetch_factor are only valid when num_workers > 0.
        # Removing them here if num_workers is 0 to avoid StatefulDataLoader errors
        if kwargs.get("num_workers", 0) == 0:
            kwargs.pop("persistent_workers", None)
            kwargs.pop("prefetch_factor", None)

    def state_dict(self) -> dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions.
        return {
            # We don't have to use pickle as DCP will serialize the state_dict. However,
            # we have to keep this for backward compatibility.
            self._rank_id: pickle.dumps(super().state_dict()),
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
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))
