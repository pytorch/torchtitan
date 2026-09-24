# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchInductor-compiled gated RMSNorm override."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from torchtitan.config import derive, override
from torchtitan.distributed.compile import maybe_regional_inductor
from torchtitan.models.common.norm import GatedRMSNorm


__all__ = [
    "CompiledGatedRMSNorm",
    "compiled_gated_rmsnorm",
]


class CompiledGatedRMSNorm(GatedRMSNorm):
    """RMSNorm with a compiled configurable unary output gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(GatedRMSNorm.Config):
        pass

    _has_inductor_region: ClassVar[bool] = True
    inductor_options: ClassVar[dict[str, bool]] = {
        "wrap_inductor_compiled_regions": True,
        "triton.cudagraphs": False,
    }

    @torch.compile(
        backend="inductor",
        fullgraph=True,
        options=inductor_options,
    )
    def _compiled_gated_rms_norm(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return GatedRMSNorm.forward(self, x, gate)

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        with maybe_regional_inductor(self.inductor_options):
            return self._compiled_gated_rms_norm(x, gate)


@override(
    target=GatedRMSNorm.Config,
    exact=True,
    description="Compile gated RMSNorm with TorchInductor.",
)
def compiled_gated_rmsnorm(
    cfg: GatedRMSNorm.Config,
) -> CompiledGatedRMSNorm.Config:
    sharding_config = cfg.sharding_config
    if sharding_config is not None:
        sharding_config = replace(
            sharding_config,
            local_spmd=True,
        )
    return derive(
        cfg,
        CompiledGatedRMSNorm.Config,
        sharding_config=sharding_config,
    )
