# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dist-MoE specialization of the shared quantized FSDP tensor lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.components.quantization._fsdp_tensor import _ShardedFSDPTensor

from dist_moe import (
    BlockScaledFormat,
    DistMoeBlockScaledConfig,
    DistMoePreparedWeight,
    prepare_blockscaled_weight,
)


__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _DistMoeMXFP8Operands:
    """Independent MXFP8 tensors owned by one FSDP unshard lifetime."""

    qdata: torch.Tensor
    fprop_scale: torch.Tensor
    dgrad_scale: torch.Tensor
    quantization_workspace: torch.Tensor

    def prepared(self, source: torch.Tensor):
        """Return the annex facade consumed by one Dist-MoE invocation."""
        return DistMoePreparedWeight(
            source=source,
            format=BlockScaledFormat.MXFP8_E4M3,
            fprop_data=self.qdata,
            fprop_scale=self.fprop_scale,
            dgrad_data=self.qdata,
            dgrad_scale=self.dgrad_scale,
            _quantization_workspace=self.quantization_workspace,
        )


def _prepare_mxfp8_weight(
    weight_EOI: torch.Tensor,
    out: _DistMoeMXFP8Operands | None = None,
) -> _DistMoeMXFP8Operands:
    """Allocate or refill the annex's grouped 32x32 MXFP8 weight operands."""
    prepared_out = None if out is None else out.prepared(weight_EOI)
    prepared = prepare_blockscaled_weight(
        weight_EOI,
        DistMoeBlockScaledConfig(),
        out=prepared_out,
    )
    if prepared.dgrad_data is not prepared.fprop_data:
        raise RuntimeError("MXFP8 Dist-MoE FPROP and DGRAD must share qdata")
    if prepared.dgrad_scale is None or prepared._quantization_workspace is None:
        raise RuntimeError("MXFP8 Dist-MoE preparation returned incomplete operands")
    return _DistMoeMXFP8Operands(
        qdata=prepared.fprop_data,
        fprop_scale=prepared.fprop_scale,
        dgrad_scale=prepared.dgrad_scale,
        quantization_workspace=prepared._quantization_workspace,
    )


class _DistMoeW13ShardedTensor(_ShardedFSDPTensor):
    """Persistent W13 parameter that prepares grouped MXFP8 compute operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        weight_E_2F_D = logical_tensor.flatten(1, 2)
        return _prepare_mxfp8_weight(weight_E_2F_D, out)


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
):
    """Prepare a weight when no FSDP implementation owns its unshard lifetime."""
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
