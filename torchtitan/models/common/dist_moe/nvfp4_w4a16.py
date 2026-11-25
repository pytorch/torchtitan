# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 weight-only (W4A16) Dist-MoE routed experts.

The weights follow dist_moe's weight-only NVFP4 recipe: Four Over Six block
scales and one global scale per expert, as in the miles/humans& W4A16 RL
recipe. Training passes the quantize-dequantized (QDQ) expert weights, prepared
once per FSDP unshard, to a weight-only Dist-MoE training context, which runs
its BF16 kernels on them and passes the weight gradients straight through.
With ``mega_fused_inference`` (RL generator), the experts keep dist_moe's NVFP4
operands of the same codes and scales and run its fused W4A16 Mega kernel,
which applies the global scale to the FP32 accumulator instead.

Shape suffixes: ``E`` local experts, ``N`` output features, ``K`` input
features.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import spmd_types as spmd
import torch

from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.quantization._fsdp_tensor import (
    _ShardedFSDPTensor,
    _UnshardedFSDPTensor,
)

from . import _dist_moe as dist_moe
from .routed_experts import DistMoeRoutedExperts


__all__ = ["NVFP4W4A16DistMoeRoutedExperts"]


def _weight_only_nvfp4() -> dist_moe.BlockScaledConfig:
    """Return dist_moe's weight-only NVFP4 (W4A16) Mega policy."""
    return dist_moe.BlockScaledConfig(
        format=dist_moe.BlockScaledFormat.NVFP4, pipeline="mega", weight_only=True
    )


@dataclass(frozen=True, slots=True)
class _NVFP4QDQOperands:
    """QDQ expert weight owned by one FSDP unshard lifetime."""

    weight: torch.Tensor

    def prepared(self, source: torch.Tensor) -> dist_moe.PreparedWeight:
        """Return the annex facade of the QDQ weight for ``source`` [E, N, K]."""
        # TODO: use a public dist_moe constructor once it has one; MXFP8 calls
        # this private one too.
        return dist_moe.PreparedWeight._create(
            source=source,
            format=dist_moe.BlockScaledFormat.NVFP4,
            fprop_data=self.weight.view(source.shape),
            fprop_scale=None,
            dgrad_data=None,
            dgrad_scale=None,
            fake_quantized=True,
        )


class _NVFP4QDQShardedTensor(_ShardedFSDPTensor):
    """Expert weight whose unsharded operand is its NVFP4 QDQ.

    W13 ``(E, 2, F, D)`` is quantized as ``(E, 2F, D)``, so gate and up share
    one global scale per expert. Blocks run along the input dimension.
    """

    def _build_operands(
        self, logical_tensor: torch.Tensor, out: _NVFP4QDQOperands | None = None
    ) -> _NVFP4QDQOperands:
        weight_ENK = logical_tensor.flatten(1, -2)
        prepared = dist_moe.prepare_block_scaled_weight(
            weight_ENK,
            _weight_only_nvfp4(),
            out=None if out is None else out.prepared(weight_ENK),
        )
        if out is None:
            return _NVFP4QDQOperands(prepared.fprop_data.view_as(logical_tensor))
        # FSDP checks that a refill keeps the operand objects, not that
        # dist_moe wrote into them; a refill elsewhere would leave stale weights.
        if prepared.fprop_data.data_ptr() != out.weight.data_ptr():
            raise RuntimeError("dist_moe did not refill the retained QDQ weight")
        return out


@dataclass(frozen=True, slots=True)
class _NVFP4MegaOperands:
    """dist_moe's NVFP4 Mega FPROP operands owned by one FSDP unshard lifetime."""

    qdata: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    global_scale_inv: torch.Tensor

    def prepared(
        self, source: torch.Tensor, *, gate_up: bool
    ) -> dist_moe.PreparedWeight:
        """Return the annex facade consumed by one Dist-MoE invocation."""
        # TODO: use a public dist_moe constructor once it has one.
        return dist_moe.PreparedWeight._create(
            source=source,
            format=dist_moe.BlockScaledFormat.NVFP4,
            fprop_data=self.qdata,
            fprop_scale=self.scale,
            dgrad_data=None,
            dgrad_scale=None,
            global_scale=self.global_scale,
            global_scale_inv=self.global_scale_inv,
            interleaved_fc13=gate_up,
        )


class _NVFP4MegaShardedTensor(_ShardedFSDPTensor):
    """Expert weight whose unsharded operands are dist_moe's NVFP4 Mega operands.

    W13, the 4-D ``(E, 2, F, D)`` weight, gets the gate/up row interleave the
    kernel's fused SwiGLU reads. A refill copies into the retained operands, so
    CUDA graphs keep reading the same addresses.
    """

    def _build_operands(
        self, logical_tensor: torch.Tensor, out: _NVFP4MegaOperands | None = None
    ) -> _NVFP4MegaOperands:
        # TODO: refill in place once dist_moe's prepare_block_scaled_weight
        # accepts out= for NVFP4 inference.
        prepared = dist_moe.prepare_block_scaled_weight(
            logical_tensor.flatten(1, -2).contiguous(),
            _weight_only_nvfp4(),
            inference=True,
            fuse_swiglu=logical_tensor.ndim == 4,
        )
        operands = _NVFP4MegaOperands(
            prepared.fprop_data,
            prepared.fprop_scale,
            prepared.global_scale,
            prepared.global_scale_inv,
        )
        if out is None:
            return operands
        for field in fields(out):
            getattr(out, field.name).copy_(getattr(operands, field.name))
        return out


def _operands(weight: torch.Tensor) -> Any:
    """Return the operands of this unshard lifetime, or build them without FSDP."""
    if isinstance(weight, _UnshardedFSDPTensor):
        return weight.operands
    assert isinstance(weight, _ShardedFSDPTensor)
    with torch.no_grad():
        return weight._build_operands(weight._tensor)


class NVFP4W4A16DistMoeRoutedExperts(DistMoeRoutedExperts):
    """Dist-MoE routed experts with NVFP4 weight-only (W4A16) quantization.

    The parameter, FSDP and checkpoint layout is the BF16 experts' one.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Configure NVFP4 W4A16 Dist-MoE routed experts.

        Args:
            mega_fused_inference: Run dist_moe's fused W4A16 Mega kernel on
                NVFP4 operands, for an inference runtime only.
        """

        mega_fused_inference: bool = False

        def __post_init__(self) -> None:
            """Validate the common expert policies and the NVFP4 block size."""
            DistMoeRoutedExperts.Config.__post_init__(self)
            if self.mega_fused_inference and (
                self.activation != "swiglu" or self.bf16_grouped_gemm_preset is not None
            ):
                raise ValueError("NVFP4 W4A16 Mega needs plain SwiGLU and no preset")
            # NVFP4 blocks are 1x16 along the input dimension
            if self.w13.in_features % 16 or self.w2.in_features % 16:
                raise ValueError("NVFP4 needs w13 and w2 in_features divisible by 16")

    def __init__(self, config: Config):
        super().__init__(config)
        self.mega_fused_inference = config.mega_fused_inference
        self.block_scaled_config = _weight_only_nvfp4()
        self._wrap_weights()

    def _wrap_weights(self) -> None:
        """Wrap W13 and W2 in the FSDP tensor type that builds their operands."""
        wrapper_cls = (
            _NVFP4MegaShardedTensor
            if self.mega_fused_inference
            else _NVFP4QDQShardedTensor
        )
        for proj in (self.w13, self.w2):
            if not isinstance(proj.weight, wrapper_cls):
                proj.weight = torch.nn.Parameter(
                    wrapper_cls(proj.weight.data),
                    requires_grad=proj.weight.requires_grad,
                )

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        # spmd_types returns a plain tensor when EP/TP shards a weight. Restore
        # the FSDP extension wrapper before fully_shard() consumes it.
        super()._parallelize(parallelism_context)
        distributed_weights = (self.w13.weight, self.w2.weight)
        self._wrap_weights()
        for proj, weight in zip((self.w13, self.w2), distributed_weights, strict=True):
            spmd.assert_type_like(proj.weight, weight)

    def _weight_operands(
        self,
    ) -> tuple[
        torch.Tensor | dist_moe.PreparedWeight,
        torch.Tensor | dist_moe.PreparedWeight,
    ]:
        """Return the Mega operands, or the QDQ weights for the training context."""
        w13_E2FD, w2_EDF = self.w13.weight, self.w2.weight
        w13_operands, w2_operands = _operands(w13_E2FD), _operands(w2_EDF)
        if self.mega_fused_inference:
            return (
                w13_operands.prepared(w13_E2FD.flatten(1, 2), gate_up=True),
                w2_operands.prepared(w2_EDF, gate_up=False),
            )
        return (
            w13_operands.prepared(w13_E2FD.flatten(1, 2)),
            w2_operands.prepared(w2_EDF),
        )
