# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
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
            random.getstate(),
            np.random.get_state(),
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
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.accelerator is not None:
            torch.get_device_module(self.accelerator).set_rng_state(
                self.accelerator_state
            )
