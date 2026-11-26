# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful

RNG_STATE = "rng_state"


@dataclass
class RngSnapshot:
    """Global streams in this process; dataset-owned streams stay with the loader."""

    cpu: torch.Tensor
    accelerator: str | None
    accelerator_state: torch.Tensor | None

    @staticmethod
    def initialized_accelerator() -> str | None:
        device = torch.accelerator.current_accelerator()
        if device is None:
            return None
        module = torch.get_device_module(device)
        # Do not initialize a device merely to checkpoint the CPU streams.
        if not getattr(module, "is_initialized", lambda: False)():
            return None
        return device.type

    @classmethod
    def capture(cls) -> "RngSnapshot":
        accelerator = cls.initialized_accelerator()
        return cls(
            torch.get_rng_state(),
            accelerator,
            torch.get_device_module(accelerator).get_rng_state()
            if accelerator is not None
            else None,
        )

    def compatible(self) -> bool:
        return self.accelerator == self.initialized_accelerator()

    def restore(self) -> None:
        if not self.compatible():
            raise ValueError("RNG accelerator backend changed since capture")
        torch.set_rng_state(self.cpu)
        if self.accelerator is not None:
            torch.get_device_module(self.accelerator).set_rng_state(
                self.accelerator_state
            )


class RngState(Stateful):
    """Stage rank-local RNG state for explicit restoration after other state loads."""

    def __init__(self) -> None:
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.pending: RngSnapshot | None = None

    def state_dict(self) -> dict[str, Any]:
        # Serialize on the calling thread, before asynchronous staging starts.
        return {
            "world_size": self.world_size,
            f"rank_{self.rank}": pickle.dumps(RngSnapshot.capture()),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict["world_size"] != self.world_size:
            raise ValueError(
                f"Checkpoint RNG world size {state_dict['world_size']} differs from "
                f"the current world size {self.world_size}. To resume without RNG "
                'restoration, set exclude_from_loading=["rng_state"].'
            )
        self.pending = pickle.loads(state_dict[f"rank_{self.rank}"])

    def restore(self) -> None:
        snapshot, self.pending = self.pending, None
        compatible = snapshot is not None and snapshot.compatible()
        if dist.is_initialized():
            # Agree on every rank before changing any training stream.
            layouts: list[Any] = [None] * self.world_size
            dist.all_gather_object(layouts, compatible)
            compatible = all(layouts)
        if not compatible:
            raise ValueError(
                "Checkpoint RNG accelerator backend differs on at least one rank. "
                'To resume without RNG restoration, set exclude_from_loading=["rng_state"].'
            )
        assert snapshot is not None
        snapshot.restore()
