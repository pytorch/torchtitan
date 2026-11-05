# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4-forward/BF16-backward prepared weights for Dist-MoE experts."""

from __future__ import annotations

from dataclasses import dataclass, field
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


__all__ = ["NVFP4FwdBF16BwdDistMoeRoutedExperts"]

# Shape suffixes: E = local experts, O = output features, I = input features,
# F = expert intermediate dimension, and D = model dimension.


def _nvfp4_policy(
    *,
    use_four_over_six: bool = False,
) -> dist_moe.BlockScaledConfig:
    """Build the Dist-MoE NVFP4-forward/BF16-backward policy."""
    return dist_moe.BlockScaledConfig(
        format=dist_moe.BlockScaledFormat.NVFP4,
        use_four_over_six=use_four_over_six,
    )


@dataclass(frozen=True, slots=True)
class _DistMoeNVFP4Operands:
    """Independent NVFP4 tensors owned by one FSDP unshard lifetime."""

    qdata: torch.Tensor
    fprop_scale: torch.Tensor
    global_scale: torch.Tensor
    global_scale_inv: torch.Tensor

    def prepared(self, source: torch.Tensor) -> dist_moe.PreparedWeight:
        """Return the Dist-MoE facade consumed by one expert invocation."""
        return dist_moe.PreparedWeight._create(
            source=source,
            format=dist_moe.BlockScaledFormat.NVFP4,
            fprop_data=self.qdata,
            fprop_scale=self.fprop_scale,
            dgrad_data=None,
            dgrad_scale=None,
            global_scale=self.global_scale,
            global_scale_inv=self.global_scale_inv,
        )


def _prepare_nvfp4_weight(
    weight_EOI: torch.Tensor,
    out: _DistMoeNVFP4Operands | None = None,
    *,
    use_four_over_six: bool,
) -> _DistMoeNVFP4Operands:
    """Allocate or refill Dist-MoE NVFP4 forward operands."""
    prepared_out = None if out is None else out.prepared(weight_EOI)
    prepared = dist_moe.prepare_block_scaled_weight(
        weight_EOI,
        _nvfp4_policy(use_four_over_six=use_four_over_six),
        out=prepared_out,
    )
    if prepared.dgrad_data is not None or prepared.dgrad_scale is not None:
        raise RuntimeError("NVFP4 Dist-MoE preparation returned DGRAD operands")
    if prepared.global_scale is None or prepared.global_scale_inv is None:
        raise RuntimeError("NVFP4 Dist-MoE preparation returned incomplete scales")
    return _DistMoeNVFP4Operands(
        qdata=prepared.fprop_data,
        fprop_scale=prepared.fprop_scale,
        global_scale=prepared.global_scale,
        global_scale_inv=prepared.global_scale_inv,
    )


class _DistMoeNVFP4ShardedTensorBase(_ShardedFSDPTensor):
    """Persistent grouped expert weight that prepares NVFP4 operands."""

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        use_four_over_six: bool = False,
        **logical_metadata: Any,
    ) -> None:
        super().__init__(tensor, **logical_metadata)
        self.use_four_over_six = use_four_over_six

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        return ["_tensor"], (self.dtype, self.use_four_over_six)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        del outer_size, outer_stride
        _, use_four_over_six = metadata
        return cls(
            inner_tensors["_tensor"],
            use_four_over_six=use_four_over_six,
        )

    def _rewrap(self, tensor: torch.Tensor) -> _DistMoeNVFP4ShardedTensorBase:
        """Wrap ``tensor`` while preserving its quantization policy."""
        return type(self)(
            tensor,
            use_four_over_six=self.use_four_over_six,
        )


class _DistMoeNVFP4W13ShardedTensor(_DistMoeNVFP4ShardedTensorBase):
    """Persistent W13 parameter that prepares grouped NVFP4 operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeNVFP4Operands | None = None,
    ) -> _DistMoeNVFP4Operands:
        return _prepare_nvfp4_weight(
            logical_tensor.flatten(1, 2),
            out,
            use_four_over_six=self.use_four_over_six,
        )


class _DistMoeNVFP4W2ShardedTensor(_DistMoeNVFP4ShardedTensorBase):
    """Persistent W2 parameter that prepares grouped NVFP4 operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeNVFP4Operands | None = None,
    ) -> _DistMoeNVFP4Operands:
        return _prepare_nvfp4_weight(
            logical_tensor,
            out,
            use_four_over_six=self.use_four_over_six,
        )


def _dynamic_prepared_weight(
    logical_weight: torch.Tensor,
    *,
    gate_up: bool,
    use_four_over_six: bool,
) -> dist_moe.PreparedWeight:
    """Prepare a weight when FSDP does not own its unshard lifetime."""
    source = logical_weight.flatten(1, 2) if gate_up else logical_weight
    storage = (
        logical_weight._tensor
        if isinstance(logical_weight, _ShardedFSDPTensor)
        else logical_weight
    )
    compute_weight = storage.flatten(1, 2) if gate_up else storage
    with torch.no_grad():
        operands = _prepare_nvfp4_weight(
            compute_weight,
            use_four_over_six=use_four_over_six,
        )
    return operands.prepared(source)


class NVFP4FwdBF16BwdDistMoeRoutedExperts(DistMoeRoutedExperts):
    """Dist-MoE experts with NVFP4 forward and BF16 STE backward."""

    block_scaled_config: dist_moe.BlockScaledConfig

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Configure cached NVFP4 forward weights and BF16 STE backward."""

        block_scaled_config: dist_moe.BlockScaledConfig = field(
            default_factory=_nvfp4_policy
        )

        def __post_init__(self) -> None:
            """Validate the common expert and NVFP4-specific policies."""
            DistMoeRoutedExperts.Config.__post_init__(self)
            if self.block_scaled_config.format is not dist_moe.BlockScaledFormat.NVFP4:
                raise ValueError(
                    "NVFP4FwdBF16BwdDistMoeRoutedExperts requires the NVFP4 format"
                )
            if self.bf16_grouped_gemm_preset is not None:
                raise ValueError(
                    "bf16_grouped_gemm_preset is not used by NVFP4 Dist-MoE"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.block_scaled_config = config.block_scaled_config
        use_four_over_six = config.block_scaled_config.use_four_over_six
        self.w13.weight = torch.nn.Parameter(
            _DistMoeNVFP4W13ShardedTensor(
                self.w13.weight.data,
                use_four_over_six=use_four_over_six,
            ),
            requires_grad=self.w13.weight.requires_grad,
        )
        self.w2.weight = torch.nn.Parameter(
            _DistMoeNVFP4W2ShardedTensor(
                self.w2.weight.data,
                use_four_over_six=use_four_over_six,
            ),
            requires_grad=self.w2.weight.requires_grad,
        )

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        # spmd_types returns a plain tensor when EP/TP shards a weight. Restore
        # the FSDP extension wrappers before fully_shard() consumes them.
        super()._parallelize(parallelism_context)
        use_four_over_six = self.block_scaled_config.use_four_over_six
        for proj, wrapper_cls in (
            (self.w13, _DistMoeNVFP4W13ShardedTensor),
            (self.w2, _DistMoeNVFP4W2ShardedTensor),
        ):
            if isinstance(proj.weight, wrapper_cls):
                continue
            distributed_weight = proj.weight
            wrapped_weight = torch.nn.Parameter(
                wrapper_cls(
                    distributed_weight.data,
                    use_four_over_six=use_four_over_six,
                ),
                requires_grad=distributed_weight.requires_grad,
            )
            spmd.assert_type_like(
                wrapped_weight,  # pyrefly: ignore [bad-argument-type]
                distributed_weight,
            )
            proj.weight = wrapped_weight

    def _weight_operands(
        self,
    ) -> tuple[
        torch.Tensor | dist_moe.PreparedWeight,
        torch.Tensor | dist_moe.PreparedWeight,
    ]:
        """Return prepared W13 and W2 operands for this unshard lifetime."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        w13_EFD = w13_E2FD.flatten(1, 2)
        use_four_over_six = self.block_scaled_config.use_four_over_six
        w13_operand = (
            w13_E2FD.operands.prepared(w13_EFD)
            if isinstance(w13_E2FD, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(
                w13_E2FD,
                gate_up=True,
                use_four_over_six=use_four_over_six,
            )
        )
        w2_operand = (
            w2_EDF.operands.prepared(w2_EDF)
            if isinstance(w2_EDF, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(
                w2_EDF,
                gate_up=False,
                use_four_over_six=use_four_over_six,
            )
        )
        return w13_operand, w2_operand
