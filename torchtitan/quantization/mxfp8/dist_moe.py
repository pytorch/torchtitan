# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 prepared-weight lifecycle for DistMoE grouped projections."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# DistMoE is an optional dependency and this module is imported only by the
# DistMoE adapter.
from dist_moe import (
    DistMoeBlockScaledConfig,
    DistMoeBlockScaledFormat,
    DistMoePreparedWeight,
    prepare_block_scaled_weight,
)

from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor


__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _DistMoeMXFP8Operands:
    """Independent MXFP8 tensors owned by one FSDP unshard lifetime."""

    qdata: torch.Tensor
    fprop_scale: torch.Tensor
    dgrad_scale: torch.Tensor

    @property
    def fprop_data(self) -> torch.Tensor:
        return self.qdata

    @property
    def dgrad_data(self) -> torch.Tensor:
        return self.qdata

    def prepared(self, source: torch.Tensor) -> DistMoePreparedWeight:
        """Rebuild the facade over storage owned by the FSDP lifecycle."""
        # Generic FSDP owns each tensor field independently and reconstructs
        # this operands dataclass after reshard. The public factory cannot wrap
        # those existing fields without requantizing, so rebuild only its
        # validated facade here and keep all storage under FSDP ownership.
        return DistMoePreparedWeight._create(
            source=source,
            format=DistMoeBlockScaledFormat.MXFP8_E4M3,
            fprop_data=self.fprop_data,
            fprop_scale=self.fprop_scale,
            dgrad_data=self.dgrad_data,
            dgrad_scale=self.dgrad_scale,
        )


def _prepare_mxfp8_weight(
    weight_EOI: torch.Tensor,
    out: _DistMoeMXFP8Operands | None = None,
) -> _DistMoeMXFP8Operands:
    """Allocate or refill the annex's grouped 32x32 MXFP8 weight operands."""
    prepared_out = None if out is None else out.prepared(weight_EOI)
    prepared = prepare_block_scaled_weight(
        weight_EOI,
        DistMoeBlockScaledConfig(),
        out=prepared_out,
    )
    if prepared.dgrad_data is None or prepared.dgrad_scale is None:
        raise RuntimeError("MXFP8 DistMoE preparation returned incomplete operands")
    if prepared.dgrad_data is not prepared.fprop_data:
        raise RuntimeError("MXFP8 DistMoE FPROP and DGRAD must share qdata")
    return _DistMoeMXFP8Operands(
        qdata=prepared.fprop_data,
        fprop_scale=prepared.fprop_scale,
        dgrad_scale=prepared.dgrad_scale,
    )


class _DistMoeW13ShardedTensor(_ShardedFSDPTensor):
    """Persistent W13 parameter that prepares grouped MXFP8 compute operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        return _prepare_mxfp8_weight(logical_tensor.flatten(1, 2), out)


class _DistMoeW2ShardedTensor(_ShardedFSDPTensor):
    """Persistent W2 parameter that prepares grouped MXFP8 compute operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        return _prepare_mxfp8_weight(logical_tensor, out)


def _dynamic_prepared_weight(
    logical_weight: torch.Tensor,
    *,
    gate_up: bool,
) -> DistMoePreparedWeight:
    """Prepare a weight when FSDP does not own its unshard lifetime."""
    source = logical_weight.flatten(1, 2) if gate_up else logical_weight
    storage = (
        logical_weight._tensor
        if isinstance(logical_weight, _ShardedFSDPTensor)
        else logical_weight
    )
    compute_weight = storage.flatten(1, 2) if gate_up else storage
    with torch.no_grad():
        operands = _prepare_mxfp8_weight(compute_weight)
    return operands.prepared(source)
