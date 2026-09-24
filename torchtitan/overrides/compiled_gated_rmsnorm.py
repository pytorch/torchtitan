# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchInductor-compiled gated RMSNorm override."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from torchtitan.config import derive, override
from torchtitan.distributed.compile import maybe_regional_inductor
from torchtitan.models.common.nn_modules import GatedRMSNorm
from torchtitan.models.kimi_k3.kda import KimiGatedRMSNorm


__all__ = [
    "CompiledGatedRMSNorm",
    "compiled_kimi_gated_rmsnorm",
]


class CompiledGatedRMSNorm(GatedRMSNorm):
    """RMSNorm with a compiled configurable unary output gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(GatedRMSNorm.Config):
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid

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
        x_THV: torch.Tensor,
        gate_THV: torch.Tensor,
    ) -> torch.Tensor:
        return GatedRMSNorm.forward(self, x_THV, gate_THV)

    def forward(
        self,
        x_THV: torch.Tensor,
        gate_THV: torch.Tensor,
    ) -> torch.Tensor:
        with maybe_regional_inductor(self.inductor_options):
            return self._compiled_gated_rms_norm(x_THV, gate_THV)


@override(
    target=KimiGatedRMSNorm.Config,
    exact=True,
    description="Compile Kimi K3 gated RMSNorm with TorchInductor.",
)
def compiled_kimi_gated_rmsnorm(
    cfg: KimiGatedRMSNorm.Config,
    *,
    activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
) -> CompiledGatedRMSNorm.Config:
    sharding_config = cfg.sharding_config
    if sharding_config is not None:
        input_shardings = (
            sharding_config.in_dst_shardings or sharding_config.in_src_shardings or {}
        )
        x_sharding = input_shardings.get("x_THV")
        gate_sharding = input_shardings.get("gate_THV")
        output_sharding = (
            sharding_config.out_src_shardings or sharding_config.out_dst_shardings
        )
        weight_sharding = sharding_config.state_shardings.get("weight")
        if x_sharding is None or gate_sharding is None or output_sharding is None:
            raise ValueError(
                "CompiledGatedRMSNorm requires input and output sharding "
                "contracts when a sharding config is present"
            )
        if weight_sharding is None:
            raise ValueError("CompiledGatedRMSNorm requires a weight sharding contract")
        sharding_config = replace(
            sharding_config,
            local_spmd=True,
        )
    return derive(
        cfg,
        CompiledGatedRMSNorm.Config,
        activation_fn=activation_fn,
        sharding_config=sharding_config,
    )
