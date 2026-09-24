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
import torch.nn.functional as F

from torchtitan.config import derive, override
from torchtitan.distributed.compile import maybe_regional_inductor
from torchtitan.models.common.decoder_sharding import attention_activation_placement
from torchtitan.models.common.nn_modules import GatedRMSNorm
from torchtitan.models.kimi_k3.kda import KimiGatedRMSNorm
from torchtitan.models.qwen3_5.gdn import Qwen35GatedRMSNorm


__all__ = [
    "CompiledGatedRMSNorm",
    "compiled_kimi_gated_rmsnorm",
    "compiled_qwen35_gated_rmsnorm",
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
    target=KimiGatedRMSNorm.Config,
    exact=True,
    description="Compile Kimi K3 gated RMSNorm with TorchInductor.",
)
def compiled_kimi_gated_rmsnorm(
    cfg: KimiGatedRMSNorm.Config,
) -> CompiledGatedRMSNorm.Config:
    sharding_config = cfg.sharding_config
    if sharding_config is not None:
        if sharding_config.state_shardings.get("weight") is None:
            raise ValueError(
                "Compiled KimiGatedRMSNorm requires a weight sharding contract"
            )
        activation = attention_activation_placement()
        input_shardings = {
            "x": activation,
            "gate": activation,
        }
        sharding_config = replace(
            sharding_config,
            in_src_shardings=input_shardings,
            in_dst_shardings=input_shardings,
            out_src_shardings=activation,
            out_dst_shardings=activation,
            local_spmd=True,
        )
    return derive(
        cfg,
        CompiledGatedRMSNorm.Config,
        activation_fn=torch.sigmoid,
        sharding_config=sharding_config,
    )


@override(
    target=Qwen35GatedRMSNorm.Config,
    exact=True,
    description="Compile Qwen3.5 gated RMSNorm with TorchInductor.",
)
def compiled_qwen35_gated_rmsnorm(
    cfg: Qwen35GatedRMSNorm.Config,
) -> CompiledGatedRMSNorm.Config:
    sharding_config = cfg.sharding_config
    if sharding_config is not None:
        if sharding_config.state_shardings.get("weight") is None:
            raise ValueError(
                "Compiled Qwen35GatedRMSNorm requires a weight sharding contract"
            )
        activation = attention_activation_placement()
        input_shardings = {
            "x": activation,
            "gate": activation,
        }
        sharding_config = replace(
            sharding_config,
            in_src_shardings=input_shardings,
            in_dst_shardings=input_shardings,
            out_src_shardings=activation,
            out_dst_shardings=activation,
            local_spmd=True,
        )
    return derive(
        cfg,
        CompiledGatedRMSNorm.Config,
        activation_fn=F.silu,
        sharding_config=sharding_config,
    )
