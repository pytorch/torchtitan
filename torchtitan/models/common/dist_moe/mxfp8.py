# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 prepared-weight lifecycle for Dist-MoE grouped projections."""

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


__all__ = ["MXFP8DistMoeRoutedExperts"]


@dataclass(frozen=True, slots=True)
class _DistMoeMXFP8Operands:
    """Independent MXFP8 tensors owned by one FSDP unshard lifetime.

    ``dgrad_scale`` is for training only. For inference, ``dgrad_scale`` is
    an empty tensor since every field is FSDP-managed and cannot be ``None``.
    """

    qdata: torch.Tensor
    fprop_scale: torch.Tensor
    dgrad_scale: torch.Tensor

    def prepared(
        self, source: torch.Tensor, *, gate_up: bool = False, inference: bool = False
    ) -> dist_moe.PreparedWeight:
        """Return the annex facade consumed by one Dist-MoE invocation."""
        if inference != (self.dgrad_scale.numel() == 0):
            raise RuntimeError(
                f"inference={inference} but dgrad_scale has "
                f"{self.dgrad_scale.numel()} elements; inference operands must have "
                "empty DGRAD scales and training operands non-empty ones"
            )
        return dist_moe.PreparedWeight._create(
            source=source,
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
            fprop_data=self.qdata,
            fprop_scale=self.fprop_scale,
            dgrad_data=None if inference else self.qdata,
            dgrad_scale=None if inference else self.dgrad_scale,
            # For inference, only W13 is interleaved and fuses SwiGLU
            interleaved_fc13=inference and gate_up,
        )


def _prepare_mxfp8_weight(
    weight_EOI: torch.Tensor,
    out: _DistMoeMXFP8Operands | None = None,
    *,
    gate_up: bool,
    inference: bool,
) -> _DistMoeMXFP8Operands:
    """Allocate or refill the annex's grouped MXFP8 weight operands.

    Training uses 32x32 block scales for both FPROP and DGRAD, which share
    quantized data.

    Inference also uses 32x32 block scales for FPROP only and has no DGRAD
    operands. For FPROP, W13's gate/up rows are additionally interleaved
    so the Mega kernel can fuse SwiGLU into the W13 epilogue.

    ``out`` holds the existing FSDP-managed operands, which are refilled in
    place so their tensor objects and addresses stay fixed (e.g. for CUDA
    graph replay).
    """
    if inference:
        # 32x32 blocks keep the trainer's quantized weights. W13 must be
        # interleaved for SwiGLU to be fused into its epilogue
        # TODO: thread BlockScaledConfig from the transform itself
        prepared = dist_moe.prepare_block_scaled_weight(
            weight_EOI,
            dist_moe.BlockScaledConfig(pipeline="mega"),
            inference=True,
            fuse_swiglu=gate_up,
            weight_block_size="32x32",
        )
        qdata, fprop_scale = prepared.fprop_data, prepared.fprop_scale

        if out is not None:
            # Refill manually since the annex only supports this for training
            out.qdata.copy_(qdata)
            out.fprop_scale.copy_(fprop_scale)
            return out
        else:
            # Use empty tensor instead of None since this is FSDP-managed
            dgrad_scale = fprop_scale.new_empty((0,))
            return _DistMoeMXFP8Operands(qdata, fprop_scale, dgrad_scale)

    # Training path: rely on the annex to refill operands through out=
    prepared_out = None if out is None else out.prepared(weight_EOI)
    prepared = dist_moe.prepare_block_scaled_weight(
        weight_EOI,
        dist_moe.BlockScaledConfig(),
        out=prepared_out,
    )
    if prepared.dgrad_data is not prepared.fprop_data:
        raise RuntimeError("MXFP8 Dist-MoE FPROP and DGRAD must share qdata")
    if prepared.dgrad_scale is None:
        raise RuntimeError("MXFP8 Dist-MoE preparation returned incomplete operands")
    return _DistMoeMXFP8Operands(
        qdata=prepared.fprop_data,
        fprop_scale=prepared.fprop_scale,
        dgrad_scale=prepared.dgrad_scale,
    )


class _DistMoeMXFP8ShardedTensorBase(_ShardedFSDPTensor):
    """Base persistent grouped expert weight that prepares MXFP8 operands.

    Subclasses implement ``_build_operands`` for W13 or W2.

    Args:
        mega_fused_inference: Prepare expert weights for the generator's
            fused SwiGLU Mega inference kernel (see
            ``DistMoeTransform.mxfp8_mega_fused_inference``).
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        mega_fused_inference: bool = False,
        **logical_metadata,
    ) -> None:
        super().__init__(tensor, **logical_metadata)
        self.mega_fused_inference = mega_fused_inference

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        return ["_tensor"], (self.dtype, self.mega_fused_inference)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        del outer_size, outer_stride
        _, mega_fused_inference = metadata
        return cls(
            inner_tensors["_tensor"],
            mega_fused_inference=mega_fused_inference,
        )

    def _rewrap(self, tensor: torch.Tensor) -> _DistMoeMXFP8ShardedTensorBase:
        """Wrap ``tensor`` like this parameter, keeping its weight format.

        FSDP ops (e.g. ``slice``, ``copy_``) rebuild the wrapper through this,
        and the base version would reset ``mega_fused_inference``.
        """
        return type(self)(tensor, mega_fused_inference=self.mega_fused_inference)


class _DistMoeW13ShardedTensor(_DistMoeMXFP8ShardedTensorBase):
    """Persistent W13 parameter that prepares grouped MXFP8 compute operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        return _prepare_mxfp8_weight(
            logical_tensor.flatten(1, 2),
            out,
            gate_up=True,
            inference=self.mega_fused_inference,
        )


class _DistMoeW2ShardedTensor(_DistMoeMXFP8ShardedTensorBase):
    """Persistent W2 parameter that prepares grouped MXFP8 compute operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        return _prepare_mxfp8_weight(
            logical_tensor,
            out,
            gate_up=False,
            inference=self.mega_fused_inference,
        )


def _dynamic_prepared_weight(
    logical_weight: torch.Tensor,
    *,
    gate_up: bool,
    inference: bool,
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
        operands = _prepare_mxfp8_weight(
            compute_weight, gate_up=gate_up, inference=inference
        )
    return operands.prepared(source, gate_up=gate_up, inference=inference)


class MXFP8DistMoeRoutedExperts(DistMoeRoutedExperts):
    """Dist-MoE routed experts using native asynchronous MXFP8 kernels.

    The module preserves the inherited W13/W2 parameter and checkpoint layout,
    while TorchTitan's shared FSDP tensor lifecycle prepares the qdata and scale
    layouts required by Dist-MoE for each unshard lifetime. Without FSDP, the
    same layouts are prepared directly from the live parameters before use.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Configure native MXFP8 execution for Dist-MoE routed experts.

        Args:
            block_scaled_config: Annex MXFP8 kernel policy. Its ``pipeline``
                selects separate staged kernels or the fused Mega pipeline;
                ``fast_math`` enables approximate sigmoid math in fused SwiGLU;
                and ``kernel_config`` is an expert-only CuTe tuning override.
            mega_fused_inference: Prepare expert weights for the
                generator's fused SwiGLU Mega inference kernel (see
                ``DistMoeTransform.mxfp8_mega_fused_inference``).
        """

        block_scaled_config: dist_moe.BlockScaledConfig = field(
            default_factory=lambda: dist_moe.BlockScaledConfig()
        )
        mega_fused_inference: bool = False

        def __post_init__(self) -> None:
            """Validate the common expert and MXFP8-specific policies."""
            DistMoeRoutedExperts.Config.__post_init__(self)
            if (
                self.block_scaled_config.format
                is not dist_moe.BlockScaledFormat.MXFP8_E4M3
            ):
                raise ValueError(
                    "MXFP8DistMoeRoutedExperts requires the MXFP8_E4M3 format"
                )
            if self.bf16_grouped_gemm_preset is not None:
                raise ValueError(
                    "bf16_grouped_gemm_preset is not used by MXFP8 Dist-MoE"
                )
            if (
                self.mega_fused_inference
                and self.block_scaled_config.pipeline != "mega"
            ):
                raise ValueError(
                    "mega_fused_inference requires "
                    "block_scaled_config.pipeline='mega'"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.block_scaled_config = config.block_scaled_config
        self.mega_fused_inference = config.mega_fused_inference
        self.w13.weight = torch.nn.Parameter(
            _DistMoeW13ShardedTensor(
                self.w13.weight.data,
                mega_fused_inference=self.mega_fused_inference,
            ),
            requires_grad=self.w13.weight.requires_grad,
        )
        self.w2.weight = torch.nn.Parameter(
            _DistMoeW2ShardedTensor(
                self.w2.weight.data,
                mega_fused_inference=self.mega_fused_inference,
            ),
            requires_grad=self.w2.weight.requires_grad,
        )

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        # spmd_types returns a plain tensor when EP/TP shards a weight. Restore
        # the FSDP extension wrappers before fully_shard() consumes them.
        super()._parallelize(parallelism_context)
        for proj, wrapper_cls in (
            (self.w13, _DistMoeW13ShardedTensor),
            (self.w2, _DistMoeW2ShardedTensor),
        ):
            if isinstance(proj.weight, wrapper_cls):
                continue
            distributed_weight = proj.weight
            wrapped_weight = torch.nn.Parameter(
                wrapper_cls(
                    distributed_weight.data,
                    mega_fused_inference=self.mega_fused_inference,
                ),
                requires_grad=distributed_weight.requires_grad,
            )
            spmd.assert_type_like(wrapped_weight, distributed_weight)
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
        w13_operand = (
            w13_E2FD.operands.prepared(
                w13_EFD, gate_up=True, inference=self.mega_fused_inference
            )
            if isinstance(w13_E2FD, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(
                w13_E2FD, gate_up=True, inference=self.mega_fused_inference
            )
        )
        w2_operand = (
            w2_EDF.operands.prepared(
                w2_EDF, gate_up=False, inference=self.mega_fused_inference
            )
            if isinstance(w2_EDF, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(
                w2_EDF, gate_up=False, inference=self.mega_fused_inference
            )
        )
        return w13_operand, w2_operand
