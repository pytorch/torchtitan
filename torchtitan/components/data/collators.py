# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configured conversion from dataset rows to trainer batches."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from torch.utils.data import default_collate

from torchtitan.components.data.dataset import DataRuntime
from torchtitan.components.data.packing import TextTrainingRow
from torchtitan.config import Configurable


TrainerBatch: TypeAlias = tuple[dict[str, Any], torch.Tensor]


class Collator(Configurable, ABC):
    """Configured row-to-batch conversion."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __call__(self, rows: Sequence[Any]) -> TrainerBatch:
        ...


class TextCollator(Collator):
    """Stacks fixed text rows into a trainer batch."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, runtime: DataRuntime) -> None:
        del config, runtime

    def __call__(self, rows: Sequence[TextTrainingRow]) -> TrainerBatch:
        inputs, labels = default_collate(list(rows))
        return inputs, labels
