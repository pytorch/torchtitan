# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.config.function import Function


class ActivationFn(Function[torch.Tensor]):
    """Base class for configurable two-input activation functions."""

    @dataclass(kw_only=True, slots=True)
    class Config(Function.Config):
        pass


@dataclass(frozen=True, slots=True)
class SwiGLU:
    """SwiGLU activation."""

    def __call__(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up


@dataclass(frozen=True, slots=True)
class SiTUGLU:
    """Kimi's SiTU-GLU activation, evaluated in FP32."""

    beta: float = 1.0
    linear_beta: float | None = None

    def __call__(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        input_dtype = gate.dtype
        gate = gate.float()
        up = up.float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(input_dtype)
