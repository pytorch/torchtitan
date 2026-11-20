# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class RngSnapshot:
    """Global streams in this process; dataset-owned streams stay with the loader."""

    python: tuple[Any, ...]
    numpy: tuple[Any, ...]
    cpu: torch.Tensor
    devices: dict[str, list[torch.Tensor]]

    @staticmethod
    def device_layout() -> dict[str, int]:
        # Do not initialize an accelerator just to inspect or save RNG state.
        return {
            name: module.device_count()
            for name in ("cuda", "xpu")
            if (module := getattr(torch, name)).is_initialized()
        }

    @classmethod
    def capture(cls) -> "RngSnapshot":
        return cls(
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            {
                name: getattr(torch, name).get_rng_state_all()
                for name in cls.device_layout()
            },
        )

    def compatible(self) -> bool:
        return {
            name: len(states) for name, states in self.devices.items()
        } == self.device_layout()

    def restore(self) -> None:
        if not self.compatible():
            raise ValueError("RNG accelerator layout changed since capture")
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        for name, states in self.devices.items():
            getattr(torch, name).set_rng_state_all(states)


@contextmanager
def preserved_rng() -> Iterator[None]:
    """Keep validation from advancing the training process's global streams."""
    snapshot = RngSnapshot.capture()
    try:
        yield
    finally:
        snapshot.restore()
