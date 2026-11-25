# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Override for the Qwen3.8-Flash-Next gated residual read.

``GatedResidual.forward`` normalizes each residual stream (``hc_norm``), projects
the normalized stream to ``down [T, R]`` and ``inject [T, S]``, and reads the
streams through a sigmoid gate built from ``down``. Two changes, same math:

- ``hc_norm`` runs as Triton kernels with a hand-written backward (the compiled
  backward fuses the input gradient into the token-wise weight-gradient sum).
- ``down`` and ``inject`` run as ONE GEMM: ``inject``'s ``[C -> S]`` GEMM is
  too narrow for a fast kernel, and one GEMM also produces a single ``x_norm``
  gradient instead of two to add. The output rows are padded to a multiple of 16.

```text
x_TC -> hc_norm (Triton) -> x_norm -+-> [down | inject] (one GEMM)
                                    |      down   -> mix = silu(down / S) -> up = Wu(mix)
                                    |      inject -> write_TS = 2 sigmoid(inject / S)
                                    +-> read (compiled region, gated by up) -> mixed_TD
```

The ``up`` projection and the stream read are unchanged.

Activate with::

    --override torchtitan_recipes.overrides.flash_next_gated_residual.fused_gated_residual
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.config import derive, override
from torchtitan.models.common import Linear
from torchtitan.models.qwen3_8_flash_next.model import GatedResidual
from torchtitan.models.qwen3_8_flash_next.norm import GroupedOffsetRMSNorm

from torchtitan_recipes.overrides.flash_next_grouped_rms_norm import (
    grouped_offset_rms_norm,
)

__all__ = ["FusedGatedResidual", "fused_gated_residual"]

# Pad the [down | inject] rows (R + S = 324) to a multiple of 16 (336) so cuBLAS
# picks an aligned kernel.
_PROJ_ROW_ALIGN = 16


class FusedGatedResidual(GatedResidual):
    """``GatedResidual`` with a Triton ``hc_norm`` and one ``[down | inject]`` GEMM.

    Parameters and checkpoint keys are unchanged.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GatedResidual.Config):
        def __post_init__(self) -> None:
            GatedResidual.Config.__post_init__(self)
            # forward() reads the projection weights directly, so a Linear
            # subclass (LoRA, quantized) or a bias would be silently dropped.
            for linear in (self.input_mix_weight_down, self.block_inject_weight):
                if linear is not None and (
                    type(linear) is not Linear.Config or linear.bias
                ):
                    raise ValueError(
                        "FusedGatedResidual requires bias-free plain Linear "
                        f"projections, got {type(linear).__qualname__}"
                    )
            # forward() also reads hc_norm's weight, eps and group size directly.
            if type(self.hc_norm) is not GroupedOffsetRMSNorm.Config:
                raise ValueError(
                    "FusedGatedResidual requires a plain GroupedOffsetRMSNorm "
                    f"hc_norm, got {type(self.hc_norm).__qualname__}"
                )

    def forward(
        self, x_TC: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        down, inject = self.input_mix_weight_down, self.block_inject_weight
        x_norm_TC = grouped_offset_rms_norm(
            x_TC,
            self.hc_norm.weight,
            eps=self.hc_norm.eps,
            group_size=self.hc_norm.group_size,
        )
        remat.recompute_needs_tensor(x_norm_TC)
        rank = down.weight.shape[0]
        blocks = [down.weight] if inject is None else [down.weight, inject.weight]
        num_proj = sum(block.shape[0] for block in blocks)
        proj_weight_PC = F.pad(
            torch.cat(blocks), (0, 0, 0, -num_proj % _PROJ_ROW_ALIGN)
        )
        proj_TP = F.linear(x_norm_TC, proj_weight_PC)
        remat.recompute_needs_tensor(proj_TP)
        mix_TR = F.silu(proj_TP[:, :rank] / self.num_streams)
        remat.recompute_needs_tensor(mix_TR)
        up_TC = self.input_mix_weight_up(mix_TR)
        remat.recompute_needs_tensor(up_TC)
        mixed_TD = self.read_streams(up_TC, x_norm_TC, self.num_streams)
        if inject is None:
            return mixed_TD
        write_TS = 2 * (proj_TP[:, rank:num_proj] / self.num_streams).sigmoid()
        return mixed_TD, x_TC, write_TS


@override(
    target=GatedResidual.Config,
    exact=True,
    description=(
        "Run Flash-Next's gated residual read with a Triton hc_norm and one "
        "[down | inject] GEMM."
    ),
)
def fused_gated_residual(cfg: GatedResidual.Config) -> FusedGatedResidual.Config:
    return derive(cfg, FusedGatedResidual.Config)
