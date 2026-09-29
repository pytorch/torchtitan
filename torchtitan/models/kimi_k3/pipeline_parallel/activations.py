# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where a pipeline rank keeps what its backward needs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from torch.distributed.pipelining.schedules import _ComputationType

from torchtitan.distributed.activation_storage import ActivationStorage

_KIND = {
    _ComputationType.FORWARD: "F",
    _ComputationType.FULL_BACKWARD: "B",
    _ComputationType.BACKWARD_INPUT: "B",
    _ComputationType.BACKWARD_WEIGHT: "W",
}

ActionKey = tuple[str, int, int]


@dataclass(kw_only=True, slots=True)
class PPMemoryConfig:
    """How a pipeline rank stores the tensors its backward reads."""

    cpu_offload: Literal["none", "all"] = "none"
    """``all`` offloads every tensor autograd saves outside the model's last layer to host
    memory (``CheckpointPolicy.MUST_CPU_OFFLOAD``); the rank store's blocks stay on the device."""

    cpu_offload_budget_gib: float | None = None
    """Host memory a rank holds offloaded saves in; a save past it stays on the device."""

    cpu_offload_prefetch_n_layers: int = 1
    """Backward reads offloaded saves back this many layers ahead."""

    min_tensor_mib: int = 1
    """Saves smaller than this stay on the device."""


def compute_actions(order: list[Any]) -> list[ActionKey]:
    return [
        (_KIND[a.computation_type], a.stage_index, a.microbatch_index)
        for a in order
        if a is not None and a.computation_type in _KIND
    ]


class BackwardPrefetch:
    """Starts reading back what a stage's backward needs first when the rank's previous compute
    action begins."""

    def __init__(self, storage: ActivationStorage, actions: list[ActionKey]) -> None:
        self._storage = storage
        self._next_backward = {
            key: following[1:]
            for key, following in zip(actions, actions[1:])
            if following[0] == "B"
        }

    def begin(self, key: ActionKey) -> None:
        item = self._next_backward.get(key)
        if item is not None:
            self._storage.prefetch_first(*item)

    def end(self, key: ActionKey) -> None:
        pass
