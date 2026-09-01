# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Routed experts stored as MXFP8, for a generator fed pre-quantized weights.

The trainer quantizes the expert weights and fuses gate with up before it
publishes them (see ``mxfp8_utils``), so the parameters here are the
grouped-GEMM operands themselves. The generator holds no high-precision copy and
has no cache to invalidate. For ``E`` experts, hidden dim ``F`` and model dim
``D``, the parameters are::

    w13_qdata     (E, 2F, D)     e4m3   quant view (E, N=2F, K=D)
    w13_scales    (E, 2F, D/32)  e8m0   one scale per 32 of K
    w2_EDF_qdata  (E, D, F)      e4m3   quant view (E, N=D,  K=F)
    w2_EDF_scales (E, D, F/32)   e8m0

The scales are stored unswizzled because that is the layout that survives
resharding, and so is the layout the trainer publishes. The GEMM needs them
swizzled, so each one has a ``*_blocked`` mirror that is rebuilt in place after
every load. The mirror costs about 3% on top of the quantized weights, and in
exchange the state dict stays equal to the wire format.

Activate this on the generator instead of
``fused_swiglu.fused_grouped_experts``, never alongside it, because both claim
``GroupedExperts.Config``::

    --override.imports torchtitan.overrides.mxfp8_inference_grouped_experts.mxfp8_inference_grouped_experts
"""

import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from torchtitan.config import derive, override
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.overrides.fused_swiglu import silu_and_mul_op
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

__all__ = ["mxfp8_inference_grouped_experts"]

# MXFP8 scaling-group size, along the contracting dim.
_BLOCK_SIZE = 32
# Tiles the blocked scale layout pads to, mirroring torchao's
# ``triton_mx_block_rearrange_per_group_3d``.
_SWIZZLE_ROW_TILE = 128
_SWIZZLE_COL_TILE = 4


class MXFP8InferenceGroupedExperts(Module):
    """Compute the routed experts from weights that are already MXFP8.

    The forward mirrors ``FusedGroupedExperts.forward``, except that it
    quantizes only the activation: the weights arrive quantized and stay that
    way. This module supports the fused layout only, so gate and up must arrive
    as a single ``w13`` and the trainer must fuse them before publishing.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        from torchao.prototype.mx_formats.config import ScaleCalculationMode

        E, F, D = config.num_experts, config.hidden_dim, config.dim
        self.num_experts = E

        # Must match what the trainer quantized the weights with, so the two
        # GEMM operands agree.
        self.scale_calculation_mode = ScaleCalculationMode.RCEIL

        qdata, scales = torch.float8_e4m3fn, torch.float8_e8m0fnu
        self.w13_qdata = nn.Parameter(
            torch.empty(E, 2 * F, D, dtype=qdata), requires_grad=False
        )
        self.w13_scales = nn.Parameter(
            torch.empty(E, 2 * F, D // _BLOCK_SIZE, dtype=scales), requires_grad=False
        )
        self.w2_EDF_qdata = nn.Parameter(
            torch.empty(E, D, F, dtype=qdata), requires_grad=False
        )
        self.w2_EDF_scales = nn.Parameter(
            torch.empty(E, D, F // _BLOCK_SIZE, dtype=scales), requires_grad=False
        )

        # Filled by refresh_blocked_scales. Not parameters or buffers: they are
        # rank-local derivations, never saved or transferred, and the swizzled
        # layout has no valid placement to declare.
        self.w13_scales_blocked: torch.Tensor | None = None
        self.w2_EDF_scales_blocked: torch.Tensor | None = None

        self.register_load_state_dict_post_hook(
            lambda module, _incompatible_keys: module.refresh_blocked_scales()
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Allocate the GEMM-layout scales, sized from the sharded parameters.

        This cannot happen in ``__init__``, where the parameter shapes are still
        global. For qwen3-30b ``w13_scales`` is ``(128, 1536, 64)`` there, while
        a rank at EP=4/TP=2 holds ``(32, 768, 64)``. Reading the shape off the
        parameter at this point yields the local one, so the module never needs
        to know either parallelism degree.

        The swizzled shape is predicted rather than obtained by swizzling a
        tensor of zeros. torchao's ``triton_mx_block_rearrange_per_group_3d``
        rounds the scale rows up to a multiple of 128 and the columns up to a
        multiple of 4, then flattens those two dimensions into one. If that
        padding ever changes, the in-place ``copy_`` in
        ``refresh_blocked_scales`` raises on the shape mismatch.
        """
        for name in ("w13_scales", "w2_EDF_scales"):
            scales = getattr(self, name)
            groups, rows, cols = scales.shape
            padded_rows = math.ceil(rows / _SWIZZLE_ROW_TILE) * _SWIZZLE_ROW_TILE
            padded_cols = math.ceil(cols / _SWIZZLE_COL_TILE) * _SWIZZLE_COL_TILE
            blocked = scales.new_empty(groups, padded_rows * padded_cols)
            setattr(self, f"{name}_blocked", blocked)

    def refresh_blocked_scales(self) -> None:
        """Rebuild the GEMM-layout scales from the parameters.

        The rebuild writes in place so that the tensors keep their addresses
        across weight syncs, which is what lets a CUDA graph captured over
        ``forward`` observe an updated weight without being recaptured.
        """
        from torchao.prototype.moe_training.kernels.mxfp8 import (
            triton_mx_block_rearrange_per_group_3d as swizzle,
        )

        with torch.no_grad():
            self.w13_scales_blocked.copy_(swizzle(self.w13_scales))
            self.w2_EDF_scales_blocked.copy_(swizzle(self.w2_EDF_scales))

    def _grouped_mm(
        self,
        act_MK: torch.Tensor,
        qdata_ENK: torch.Tensor,
        scales_blocked: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        """Run one grouped GEMM, quantizing only the activation.

        This mirrors torchao's ``_compute_fwd_sm100`` with the weight-quant step
        removed, since the weight arrived quantized, and it builds no autograd
        graph. It quantizes the activation with the CuteDSL kernel because that
        is what the trainer uses under its default ``KernelPreference.AUTO``, so
        activation quantization is identical on both sides.
        """
        from torchao.prototype.moe_training.kernels.mxfp8 import (
            mxfp8_quantize_2d_1x32_cutedsl,
        )

        act_e4m3, act_scales_blocked = mxfp8_quantize_2d_1x32_cutedsl(
            act_MK,
            scaling_mode=self.scale_calculation_mode.value.lower(),
            offs=offsets_E,
        )
        return torch._scaled_grouped_mm(
            act_e4m3,
            qdata_ENK.transpose(-2, -1),  # (E, N, K) -> (E, K, N)
            act_scales_blocked,
            scales_blocked,
            offs=offsets_E,
            out_dtype=torch.bfloat16,
        )

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        # Token groups are already padded by the mxfp8 token dispatcher.
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        gate_up_R2F = self._grouped_mm(
            x_RD.bfloat16(), self.w13_qdata, self.w13_scales_blocked, offsets_E
        )
        # Take the hidden dim from the GEMM output rather than from the weight:
        # TP shards the parameter's F, so the parameter's is not this rank's.
        F = gate_up_R2F.shape[-1] // 2
        gate_RF, up_RF = gate_up_R2F.reshape(-1, F, 2).unbind(-1)
        h_RF = silu_and_mul_op(gate_RF, up_RF, offsets_E)
        return self._grouped_mm(
            h_RF, self.w2_EDF_qdata, self.w2_EDF_scales_blocked, offsets_E
        ).type_as(x_RD)


def _mxfp8_inference_sharding(base: ShardingConfig) -> ShardingConfig:
    """Map the bf16 expert shardings onto the quantized parameters.

    The quantized tensors keep their sharded axes in the same positions as the
    weights they were built from, so the layouts carry over unchanged: the
    ``w13_*`` parameters shard like ``w1_EFD`` and the ``w2_EDF_*`` parameters
    shard like ``w2_EDF``.
    """
    state = dict(base.state_shardings)
    gate_layout = state.pop("w1_EFD")
    state.pop("w3_EFD")
    down_layout = state.pop("w2_EDF")
    state["w13_qdata"] = gate_layout
    state["w13_scales"] = gate_layout
    state["w2_EDF_qdata"] = down_layout
    state["w2_EDF_scales"] = down_layout
    return replace(base, state_shardings=state)


@override(
    target=GroupedExperts.Config,
    description="Routed experts stored as MXFP8, fed pre-quantized weights by the trainer.",
)
def mxfp8_inference_grouped_experts(
    cfg: GroupedExperts.Config,
) -> GroupedExperts.Config:
    # The factory is memoized, so this is the same class the MXFP8 converter
    # produced.
    from torchtitan.components.quantization.mx import _get_mxfp8_qat_grouped_experts_cls

    MXFP8QATGroupedExperts = _get_mxfp8_qat_grouped_experts_cls(GroupedExperts)
    if type(cfg) not in (GroupedExperts.Config, MXFP8QATGroupedExperts.Config):
        return cfg

    # No meaningful initializer for quantized parameters: every value arrives
    # from the trainer. Zeroing just defines them until the first sync.
    quantized = derive(
        cfg,
        MXFP8InferenceGroupedExperts.Config,
        param_init={
            name: nn.init.zeros_
            for name in ("w13_qdata", "w13_scales", "w2_EDF_qdata", "w2_EDF_scales")
        },
    )
    base = cfg.sharding_config
    if base is not None:
        quantized.sharding_config = _mxfp8_inference_sharding(base)
    return quantized
