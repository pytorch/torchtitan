# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 weight-only (W4A16) fake quantization for Dist-MoE routed experts.

The expert weights are quantized to NVFP4 and dequantized back (QDQ) once per
FSDP unshard lifetime, by a fused Triton kernel on GPU
(``nvfp4_qdq_kernel.py``); Dist-MoE then runs its BF16 kernels on the QDQ weights.
Activations stay BF16. The trainer and the RL generator build the same module,
so both run the same QDQ weights.

With ``mega_fused_inference`` (an inference-only generator), the module instead
keeps the real NVFP4 operands -- packed E2M1 codes, blocked E4M3 scales and
per-expert global scales from dist_moe's quantizer -- and runs dist_moe's fused
W4A16 Mega kernel (``BlockScaledConfig(weight_only=True)``), which decodes them
to BF16 on chip. Its codes and scales are the ones the QDQ weights dequantize;
it differs from the QDQ path only in rounding: the kernel applies the global
scale to the FP32 accumulator instead of rounding each dequantized weight to
BF16.

Shape suffixes: ``E`` local experts, ``N`` output features, ``K`` input
features, ``B`` NVFP4 blocks along ``K``, ``V`` values in one block.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

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


# NVFP4: E2M1 values in 1x16 blocks along K, one E4M3 scale per block, and one
# FP32 global scale per expert. The constants and the operation order below
# follow dist_moe's NVFP4 weight quantizer (``nvfp4_weight_global_scale`` and
# ``_nvfp4_scale_and_recip``), so these QDQ weights are what its NVFP4 kernels
# would dequantize.
_NVFP4_BLOCK_SIZE = 16
_FP4_E2M1_MAX = 6.0
_FP8_E4M3_MAX = 448.0
_FP8_E4M3_MIN_SUBNORMAL = 2.0**-9
_NVFP4_MAX = _FP8_E4M3_MAX * _FP4_E2M1_MAX
_NVFP4_GLOBAL_SCALE_EPS = 1e-12


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 values to E2M1 with round-to-nearest-even and saturation.

    The E2M1 magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4 and 6. A magnitude exactly
    halfway between two of them rounds to the one whose code is even, so the
    ties at 0.25, 1.25, 2.5 and 5 round down and those at 0.75, 1.75 and 3.5
    round up. Magnitudes above 6 saturate to 6, like ``cvt.rn.satfinite``.
    """
    magnitude = x.abs()
    rounded = torch.where(magnitude > 5.0, 6.0, 4.0)
    rounded = torch.where(magnitude < 3.5, 3.0, rounded)
    rounded = torch.where(magnitude <= 2.5, 2.0, rounded)
    rounded = torch.where(magnitude < 1.75, 1.5, rounded)
    rounded = torch.where(magnitude <= 1.25, 1.0, rounded)
    rounded = torch.where(magnitude < 0.75, 0.5, rounded)
    rounded = torch.where(magnitude <= 0.25, 0.0, rounded)
    return torch.copysign(rounded, x)


def _nvfp4_global_scale(weight_ENK: torch.Tensor) -> torch.Tensor:
    """Return each expert's FP32 global scale, ``2688 / amax``."""
    amax_E = torch.linalg.vector_norm(
        weight_ENK, ord=float("inf"), dim=(-2, -1), dtype=torch.float32
    ).clamp(min=_NVFP4_GLOBAL_SCALE_EPS)
    # Reciprocal-then-multiply, as dist_moe computes the global scale; a
    # division can differ by 1 ulp and flip codes near rounding boundaries.
    global_scale_E = _NVFP4_MAX * torch.reciprocal(amax_E)
    return torch.where(
        global_scale_E > _NVFP4_GLOBAL_SCALE_EPS,
        global_scale_E,
        _NVFP4_GLOBAL_SCALE_EPS,
    )


def _nvfp4_qdq(
    weight_ENK: torch.Tensor, out_ENK: torch.Tensor | None = None
) -> torch.Tensor:
    """Quantize grouped expert weights to NVFP4 and dequantize them back.

    Each expert gets one global scale from its own amax, so the gate and up
    halves of a W13 expert share it. The result has ``weight_ENK``'s dtype and
    is written into ``out_ENK`` when given. Accelerator tensors run the fused
    Triton kernel; CPU tensors run ``_nvfp4_qdq_reference``, which the kernel
    matches bit for bit.
    """
    if weight_ENK.device.type == "cpu":
        result_ENK = _nvfp4_qdq_reference(weight_ENK)
        if out_ENK is None:
            return result_ENK
        return out_ENK.copy_(result_ENK)
    from .nvfp4_qdq_kernel import nvfp4_qdq_into

    global_scale_E = _nvfp4_global_scale(weight_ENK)
    if out_ENK is None:
        out_ENK = torch.empty_like(weight_ENK, memory_format=torch.contiguous_format)
    nvfp4_qdq_into(weight_ENK.contiguous(), global_scale_E, out_ENK)
    return out_ENK


def _nvfp4_qdq_reference(weight_ENK: torch.Tensor) -> torch.Tensor:
    """Eager NVFP4 QDQ in PyTorch ops: the CPU path and the kernel's oracle."""
    num_experts = weight_ENK.shape[0]
    global_scale_E11 = _nvfp4_global_scale(weight_ENK).view(num_experts, 1, 1)

    x_ENBV = weight_ENK.float().unflatten(-1, (-1, _NVFP4_BLOCK_SIZE))
    block_amax_ENB = x_ENBV.abs().amax(dim=-1)
    # The order (amax * global_scale) * (1/6) matters: it decides the E4M3
    # rounding of ties, and with it every code in the block.
    block_scale_ENB = (
        ((block_amax_ENB * global_scale_E11) * (1.0 / _FP4_E2M1_MAX))
        .clamp(min=_FP8_E4M3_MIN_SUBNORMAL, max=_FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    codes_ENBV = _round_to_e2m1(
        x_ENBV * (global_scale_E11 / block_scale_ENB).unsqueeze(-1)
    )
    # codes * block_scale is exact in FP32 (E2M1 times E4M3), so the
    # dequantized value rounds once, in the division by the global scale.
    dequant_ENBV = (
        codes_ENBV * block_scale_ENB.unsqueeze(-1) / global_scale_E11.unsqueeze(-1)
    )
    return dequant_ENBV.flatten(-2).to(weight_ENK.dtype)


@dataclass(frozen=True, slots=True)
class _NVFP4W4A16Operands:
    """QDQ expert weight owned by one FSDP unshard lifetime."""

    weight: torch.Tensor


class _NVFP4W4A16ShardedTensor(_ShardedFSDPTensor):
    """Persistent grouped expert weight whose unsharded operand is its NVFP4 QDQ.

    W13 ``(E, 2, F, D)`` is quantized as ``(E, 2F, D)``, so gate and up share
    one global scale per expert; W2 ``(E, D, F)`` is quantized as is. Blocks
    run along the last (input) dimension in both.
    """

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _NVFP4W4A16Operands | None = None,
    ) -> _NVFP4W4A16Operands:
        weight_ENK = logical_tensor.flatten(1, -2)
        if out is None:
            return _NVFP4W4A16Operands(_nvfp4_qdq(weight_ENK).view_as(logical_tensor))
        _nvfp4_qdq(weight_ENK, out.weight.flatten(1, -2))
        return out


class _StraightThroughQDQ(torch.autograd.Function):
    """Compute with the QDQ weight; pass its gradient to the parameter unchanged.

    Dist-MoE accumulates WGRAD in place only once the parameter has a gradient
    buffer. Before that it returns WGRAD to autograd for the operand, so the
    operand needs this edge to the parameter. Forward never reads ``weight``:
    an FSDP unsharded parameter has no storage.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, qdq_weight: torch.Tensor) -> torch.Tensor:
        del ctx, weight
        return qdq_weight.detach()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        del ctx
        return grad, None


def _qdq_compute_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return the QDQ weight Dist-MoE computes with, in the logical layout."""
    if isinstance(weight, _UnshardedFSDPTensor):
        qdq_weight = weight.operands.weight
    else:
        # Without FSDP there is no unshard lifetime to cache the QDQ weight in.
        storage = weight._tensor if isinstance(weight, _ShardedFSDPTensor) else weight
        with torch.no_grad():
            qdq_weight = _nvfp4_qdq(storage.flatten(1, -2)).view_as(storage)
    return _StraightThroughQDQ.apply(weight, qdq_weight)


def _mega_block_scaled_config() -> dist_moe.BlockScaledConfig:
    """Dist-MoE policy of the fused W4A16 Mega inference kernel."""
    return dist_moe.BlockScaledConfig(
        format=dist_moe.BlockScaledFormat.NVFP4, pipeline="mega", weight_only=True
    )


@dataclass(frozen=True, slots=True)
class _NVFP4MegaOperands:
    """NVFP4 FPROP operands owned by one FSDP unshard lifetime."""

    qdata: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    global_scale_inv: torch.Tensor

    def prepared(
        self, source: torch.Tensor, *, gate_up: bool
    ) -> dist_moe.PreparedWeight:
        """Return the annex facade consumed by one Dist-MoE invocation."""
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


def _prepare_nvfp4_mega_weight(
    weight_ENK: torch.Tensor,
    out: _NVFP4MegaOperands | None = None,
    *,
    gate_up: bool,
) -> _NVFP4MegaOperands:
    """Quantize grouped expert weights to the W4A16 Mega NVFP4 operands.

    W13 rows are interleaved in gate/up groups so the kernel fuses SwiGLU into
    its W13 epilogue. ``out`` holds existing FSDP-managed operands, refilled in
    place so their addresses stay fixed for CUDA graph replay.
    """
    # TODO: refill in place once dist_moe's prepare_block_scaled_weight
    # accepts out= for NVFP4 inference.
    prepared = dist_moe.prepare_block_scaled_weight(
        weight_ENK.contiguous(),
        _mega_block_scaled_config(),
        inference=True,
        fuse_swiglu=gate_up,
    )
    operands = _NVFP4MegaOperands(
        qdata=prepared.fprop_data,
        scale=prepared.fprop_scale,
        global_scale=prepared.global_scale,
        global_scale_inv=prepared.global_scale_inv,
    )
    if out is None:
        return operands
    for field in fields(_NVFP4MegaOperands):
        getattr(out, field.name).copy_(getattr(operands, field.name))
    return out


class _NVFP4MegaW13ShardedTensor(_ShardedFSDPTensor):
    """Persistent W13 ``(E, 2, F, D)`` whose unsharded operands are NVFP4."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _NVFP4MegaOperands | None = None,
    ) -> _NVFP4MegaOperands:
        return _prepare_nvfp4_mega_weight(
            logical_tensor.flatten(1, 2), out, gate_up=True
        )


class _NVFP4MegaW2ShardedTensor(_ShardedFSDPTensor):
    """Persistent W2 ``(E, D, F)`` whose unsharded operands are NVFP4."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _NVFP4MegaOperands | None = None,
    ) -> _NVFP4MegaOperands:
        return _prepare_nvfp4_mega_weight(logical_tensor, out, gate_up=False)


def _mega_prepared_weight(
    weight: torch.Tensor, *, gate_up: bool
) -> dist_moe.PreparedWeight:
    """Return the NVFP4 W13 or W2 operand for this unshard lifetime."""
    source = weight.flatten(1, 2) if gate_up else weight
    if isinstance(weight, _UnshardedFSDPTensor):
        return weight.operands.prepared(source, gate_up=gate_up)
    # Without FSDP there is no unshard lifetime to cache the operands in.
    storage = weight._tensor if isinstance(weight, _ShardedFSDPTensor) else weight
    with torch.no_grad():
        operands = _prepare_nvfp4_mega_weight(
            storage.flatten(1, 2) if gate_up else storage, gate_up=gate_up
        )
    return operands.prepared(source, gate_up=gate_up)


class NVFP4W4A16DistMoeRoutedExperts(DistMoeRoutedExperts):
    """Dist-MoE routed experts with NVFP4 weight-only (W4A16) fake quantization.

    FPROP and DGRAD run the BF16 Dist-MoE kernels on the NVFP4 QDQ weights.
    WGRAD reaches the high-precision parameters unchanged (a straight-through
    estimator). With ``mega_fused_inference``, FPROP instead runs the fused W4A16
    Mega kernel on real NVFP4 operands. The parameter and checkpoint layout is
    the BF16 one either way.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Configure NVFP4 W4A16 Dist-MoE routed experts.

        Args:
            mega_fused_inference: Run dist_moe's fused W4A16 Mega inference
                kernel (SwiGLU fused into the W13 epilogue) on real NVFP4
                operands instead of the BF16 kernels on QDQ weights. Inference
                only: dist_moe rejects it unless the runtime is an inference
                one (see ``DistMoeTransform.nvfp4_w4a16_mega_fused_inference``).
        """

        mega_fused_inference: bool = False

        def __post_init__(self) -> None:
            """Validate the common expert policies and the NVFP4 block size."""
            DistMoeRoutedExperts.Config.__post_init__(self)
            if self.mega_fused_inference:
                if self.activation != "swiglu":
                    raise ValueError("NVFP4 W4A16 Mega supports plain SwiGLU only")
                if self.bf16_grouped_gemm_preset is not None:
                    raise ValueError(
                        "bf16_grouped_gemm_preset is not used by NVFP4 W4A16 Mega"
                    )
            for name, in_features in (
                ("w13", self.w13.in_features),
                ("w2", self.w2.in_features),
            ):
                if in_features % _NVFP4_BLOCK_SIZE:
                    raise ValueError(
                        f"NVFP4 needs {name} in_features to be a multiple of "
                        f"{_NVFP4_BLOCK_SIZE}, got {in_features}"
                    )

    def __init__(self, config: Config):
        super().__init__(config)
        self.mega_fused_inference = config.mega_fused_inference
        if self.mega_fused_inference:
            self.block_scaled_config = _mega_block_scaled_config()
        for proj, wrapper_cls in self._weight_wrappers():
            proj.weight = torch.nn.Parameter(
                wrapper_cls(proj.weight.data),
                requires_grad=proj.weight.requires_grad,
            )

    def _weight_wrappers(
        self,
    ) -> tuple[tuple[torch.nn.Module, type[_ShardedFSDPTensor]], ...]:
        """Return each projection with the FSDP tensor type preparing its operands."""
        if self.mega_fused_inference:
            return (
                (self.w13, _NVFP4MegaW13ShardedTensor),
                (self.w2, _NVFP4MegaW2ShardedTensor),
            )
        return (
            (self.w13, _NVFP4W4A16ShardedTensor),
            (self.w2, _NVFP4W4A16ShardedTensor),
        )

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        # spmd_types returns a plain tensor when EP/TP shards a weight. Restore
        # the FSDP extension wrapper before fully_shard() consumes it.
        super()._parallelize(parallelism_context)
        for proj, wrapper_cls in self._weight_wrappers():
            if isinstance(proj.weight, wrapper_cls):
                continue
            distributed_weight = proj.weight
            wrapped_weight = torch.nn.Parameter(
                wrapper_cls(distributed_weight.data),
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
        """Return the W13 and W2 operands for this unshard lifetime."""
        if self.mega_fused_inference:
            return (
                _mega_prepared_weight(self.w13.weight, gate_up=True),
                _mega_prepared_weight(self.w2.weight, gate_up=False),
            )
        w13_EFD = _qdq_compute_weight(self.w13.weight).flatten(1, 2)
        w2_EDF = _qdq_compute_weight(self.w2.weight)
        return w13_EFD, w2_EDF

    def _wgrad_parameter_owners(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Point in-place WGRAD at the parameters; the QDQ operands are not views."""
        if self.mega_fused_inference or not self.inplace_wgrad_accum:
            return None
        return self.w13.weight, self.w2.weight
