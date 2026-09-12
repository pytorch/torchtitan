# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.config.configurable import Configurable
from torchtitan.config.function import Function


class ActivationFn(Function[torch.Tensor], ABC):
    """Base class for configurable two-input activation functions."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):  # pyrefly: ignore[bad-override]
        pass

    @abstractmethod
    def __call__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        *,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass


class SwiGLU(ActivationFn):
    """SwiGLU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        *,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del offsets
        return F.silu(gate) * up


class SiTUGLU(ActivationFn):
    """Kimi's SiTU-GLU activation, evaluated in FP32."""

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationFn.Config):
        beta: float = 1.0
        linear_beta: float | None = None

    def __init__(self, config: Config) -> None:
        self.beta = config.beta
        self.linear_beta = config.linear_beta

    def __call__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        *,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del offsets
        input_dtype = gate.dtype
        gate = gate.float()
        up = up.float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(input_dtype)
