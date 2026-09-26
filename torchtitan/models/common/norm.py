# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared normalization modules."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.activation import UnaryActivationFn
from torchtitan.protocols.module import Module


class GatedRMSNorm(Module):
    """Apply RMSNorm followed by a compiled unary gate activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float
        activation_fn: UnaryActivationFn.Config

    inductor_options: ClassVar[dict[str, Callable[..., Any] | bool | int | str]] = {
        "wrap_inductor_compiled_regions": True,
        "triton.cudagraphs": False,
    }

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.activation_fn = config.activation_fn.build()
        self.weight = nn.Parameter(torch.empty(config.dim))

    @torch.compile(
        backend="inductor",
        fullgraph=True,
        options=inductor_options,
    )
    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = F.rms_norm(
            x.float(),
            (x.shape[-1],),
            self.weight.float(),
            self.eps,
        )
        return (normalized * self.activation_fn(gate.float())).to(input_dtype)


__all__ = ["GatedRMSNorm"]
