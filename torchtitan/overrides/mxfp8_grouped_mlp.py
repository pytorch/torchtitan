# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Fused MXFP8 grouped-MLP routed experts over the torchao A/B/C kernels.

The composite autograd Function runs the full routed-expert SwiGLU MLP with
three physically fused torchao CuTe DSL kernels plus the existing
``torch._scaled_grouped_mm`` on prequantized operands:

* forward:  casts -> ``torchao::mxfp8_grouped_gemm_swiglu_fwd`` (kernel A:
  FC1 grouped GEMM + SwiGLU + rowwise/colwise MXFP8 quantization of the
  activation) -> ``_scaled_grouped_mm`` FC2.
* backward: casts -> ``torchao::mxfp8_grouped_gemm_dswiglu_bwd`` (kernel B:
  FC2 dgrad + dSwiGLU + dual quantization) -> ``_scaled_grouped_mm`` FC1
  dgrad -> ``torchao::mxfp8_grouped_gemm_wgrad`` (kernel C) twice.

Every cast in this module is non-CuTe (triton/torch) and bitwise-identical to
the pure-torch ``to_mx`` RCEIL reference, so this path runs on a pristine
``cute_utils`` (only the kernels A/B/C themselves are CuTe DSL). This module
must never call into ``cute_utils``.

ABI preconditions (validated by the ops at every call): ``D``, ``F`` and the
allocated row count are positive multiples of 128, and every per-expert row
count ``m[g]`` is a nonnegative multiple of 128 — the latter is only
guaranteed by a ``TorchAOTokenDispatcher`` with ``pad_multiple=128``, which is
why the override targets ``RoutedExperts.Config`` (the one node owning both
the dispatcher and the inner experts configs).

Activate with ``--override.imports
torchtitan.overrides.mxfp8_grouped_mlp.mxfp8_grouped_experts`` on a config
whose experts were converted by ``MXFP8GroupedExpertsConverter``
(``pad_multiple=128``, recipe ``mxfp8_rceil``). When any gate fails the
factory warns with the specific reason and leaves the config unchanged, so
the run falls back to the converter's unfused MXFP8 path. NOTE: a declined
run still logs ``[Override] ...`` identity lines and ``Applied N
override(s)``; the only accepted evidence that the fused path is active is
the ``torchao::`` op counts in a profiler trace.
"""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor

# Importing the ops module registers the three torchao:: custom ops; the CuTe
# DSL is only imported lazily inside the op bodies at first real launch.
from torchao.prototype.moe_training.kernels.mxfp8 import (
    triton_mx_block_rearrange_per_group_3d,
)
from torchao.prototype.moe_training.mxfp8_grouped_mlp import (
    _mxfp8_grouped_mlp_kernels_available,
    is_supported,
)
from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_dim0,
    triton_to_mxfp8_dim1,
)
from torchao.prototype.mx_formats.utils import to_blocked

from torchtitan.components.quantization.mx import _get_mxfp8_grouped_experts_cls
from torchtitan.config import derive, override
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts
from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher
from torchtitan.overrides.fused_swiglu import (
    _fuse_w13_grouped_experts_param_init,
    _fuse_w13_grouped_experts_sharding,
    FusedGroupedExperts,
)
from torchtitan.tools.logging import logger

__all__ = [
    "MXFP8FusedGroupedExperts",
    "mxfp8_fused_grouped_mlp",
    "mxfp8_grouped_experts",
]

_BLOCK_SIZE = 32
_SCALING_MODE = "rceil"
# Row probe for the config-time shape gate; the real R is runtime-validated
# by the ops on every call (128-alignment is the dispatcher's pad_multiple).
_GROUP_ALIGNMENT = 128


def _cast_rowwise(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1x32 rowwise RCEIL cast: row-major qdata + whole-matrix blocked scales."""
    qdata, scales = triton_to_mxfp8_dim0(t, _BLOCK_SIZE, _SCALING_MODE)
    return qdata, to_blocked(scales)


def _cast_colwise(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """32x1 colwise RCEIL cast of ``[R, N]``: qdata stride ``(1, R)`` +
    whole-matrix blocked scales for logical ``[N, R/32]`` (kernel C operand
    layout — never the per-group ``_K_groups`` rearrangement, which has the
    same byte count but a different block order and corrupts silently)."""
    qdata, scales = triton_to_mxfp8_dim1(t, _BLOCK_SIZE, _SCALING_MODE)
    return qdata, to_blocked(scales)


def _cast_weight_rowwise_3d(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[G, N, K]`` quantized along K: contiguous qdata + per-group blocked
    scales. Transposed views of the result are the K-major ``mat2`` operands
    of kernel A and the FC2 ``_scaled_grouped_mm``."""
    qdata, scales = triton_to_mxfp8_dim0(w, _BLOCK_SIZE, _SCALING_MODE)
    return qdata, triton_mx_block_rearrange_per_group_3d(scales)


def _cast_weight_colwise_3d(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[G, N, K]`` quantized along N: qdata stride ``(N*K, 1, N)`` +
    per-group blocked scales for logical ``[K, N/32]``.

    No fused non-CuTe kernel exists for this orientation, so it is a
    per-group ``triton_to_mxfp8_dim1`` loop (2·G extra launches per layer
    backward, accepted for v1). Stride trap: ``torch.stack`` on the
    ``(1, N)``-strided dim1 outputs silently materializes ROW-major qdata
    (values equal, stride wrong — the grouped ops reject it), so stack the
    transposed contiguous views and re-transpose.
    """
    qdatas, scales = [], []
    for g in range(w.shape[0]):
        qdata_g, scales_g = triton_to_mxfp8_dim1(w[g], _BLOCK_SIZE, _SCALING_MODE)
        qdatas.append(qdata_g.t())
        scales.append(to_blocked(scales_g))
    return torch.stack(qdatas).transpose(-2, -1), torch.stack(scales)


class _MXFP8GroupedMLP(torch.autograd.Function):
    """Composite MXFP8 grouped SwiGLU MLP (kernel A + FC2 forward; kernel B +
    FC1 dgrad + kernel C x2 backward).

    All inputs are plain BF16 CUDA tensors (the module prologue casts and
    un-DTensors them); ``dy`` arrives as contiguous BF16 ``[R, D]``. Weight
    casts are lazy: forward quantizes the rowwise views, backward requantizes
    the colwise views from the saved BF16 parameter references — safe because
    the same-step backward always precedes the optimizer update (an update in
    between trips the autograd version counter).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        G, F, _, D = w13.shape
        # Element-interleaved flat view [gate_0, up_0, ..., gate_{F-1},
        # up_{F-1}] along 2F — the layout kernels A/B produce and consume.
        w13i = w13.reshape(G, 2 * F, D)

        x_row_q, x_row_sf = _cast_rowwise(x)
        # x colwise is a forward-side cast because it is retained for the
        # FC1 wgrad (kernel C) in backward.
        x_col_q, x_col_sf = _cast_colwise(x)
        # w13 rowwise: quantize along D, then transpose to the ABI's
        # [G, D, 2F] stride (2FD, 1, D) operand of kernel A.
        w13_row_q, w13_row_sf = _cast_weight_rowwise_3d(w13i)
        z_bf16, h_row_q, h_row_sf, h_col_q, h_col_sf = (
            torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(
                x_row_q,
                x_row_sf,
                w13_row_q.transpose(-2, -1),
                w13_row_sf,
                offsets,
            )
        )
        # FC2 forward through the existing grouped GEMM on prequantized
        # operands. w2 rowwise: quantize along F, transposed to the K-major
        # [G, F, D] stride (DF, 1, F) mat2. _scaled_grouped_mm requires 2-D
        # scales, so the flat blocked buffers MUST be reshaped here.
        w2_row_q, w2_row_sf = _cast_weight_rowwise_3d(w2)
        y = torch._scaled_grouped_mm(
            h_row_q,
            w2_row_q.transpose(-2, -1),
            h_row_sf.reshape(x.shape[0], -1),
            w2_row_sf.reshape(G, -1),
            offs=offsets,
            out_dtype=torch.bfloat16,
        )
        # CONTRACT §3.1 retain list: z_bf16 + h_col + x_col + offsets, plus
        # the BF16 w13/w2 references for the lazy backward-side casts.
        ctx.save_for_backward(
            z_bf16, h_col_q, h_col_sf, x_col_q, x_col_sf, offsets, w13, w2
        )
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        z_bf16, h_col_q, h_col_sf, x_col_q, x_col_sf, offsets, w13, w2 = (
            ctx.saved_tensors
        )
        G, F, _, D = w13.shape
        w13i = w13.reshape(G, 2 * F, D)

        # The triton casts assert contiguity; dy is contiguous today (BF16
        # [R, D] stride (D, 1)) but that is a live invariant, not a given.
        dy = dy.contiguous()
        dy_row_q, dy_row_sf = _cast_rowwise(dy)
        dy_col_q, dy_col_sf = _cast_colwise(dy)
        # w2 colwise (kernel B operand): quantize along D, landing at the
        # dgrad orientation [G, D, F] stride (DF, 1, D).
        w2_col_q, w2_col_sf = _cast_weight_colwise_3d(w2)
        dz_row_q, dz_row_sf, dz_col_q, dz_col_sf = (
            torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(
                dy_row_q, dy_row_sf, w2_col_q, w2_col_sf, z_bf16, offsets
            )
        )
        # FC1 dgrad: w13 colwise quantizes along 2F, landing at the K-major
        # [G, 2F, D] stride (2FD, 1, 2F) mat2 (K = 2F).
        w13_col_q, w13_col_sf = _cast_weight_colwise_3d(w13i)
        dx = torch._scaled_grouped_mm(
            dz_row_q,
            w13_col_q,
            dz_row_sf.reshape(dy.shape[0], -1),
            w13_col_sf.reshape(G, -1),
            offs=offsets,
            out_dtype=torch.bfloat16,
        )
        # Kernel C is generic over both wgrads: dw2 comes out [G, D, F]
        # (N=D, K=F), matching w2_EDF with NO transpose; dw13 comes out
        # [G, 2F, D] (N=2F, K=D) and views back to the [G, F, 2, D] param.
        dw2 = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(
            dy_col_q, dy_col_sf, h_col_q, h_col_sf, offsets
        )
        dw13 = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(
            dz_col_q, dz_col_sf, x_col_q, x_col_sf, offsets
        )
        return dx, dw13.view(G, F, 2, D), dw2, None


def mxfp8_fused_grouped_mlp(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Fused MXFP8 grouped SwiGLU MLP: ``x [R, D] -> y [R, D]`` (BF16).

    Args:
        x: BF16 ``[R, D]`` expert-major padded rows; every per-expert group
            is a multiple of 128 rows.
        w13: BF16 ``[G, F, 2, D]`` fused gate/up weight (gate index 0).
        w2: BF16 ``[G, D, F]`` down-projection weight.
        offsets: int32 CUDA ``[G]`` exclusive per-expert end rows,
            ``offsets[-1] <= R``. Rows past ``offsets[-1]`` of ``y`` (and of
            ``dx`` in backward) are left UNWRITTEN by ``_scaled_grouped_mm``.
    """
    return _MXFP8GroupedMLP.apply(x, w13, w2, offsets)


class MXFP8FusedGroupedExperts(FusedGroupedExperts):
    """Routed experts computed by the fused MXFP8 grouped-MLP composite.

    Inherits the fused ``w13 (E, F, 2, D)`` parametrization and the
    stock-layout (``w1_EFD``/``w3_EFD``) checkpoint save/load hooks from
    :class:`FusedGroupedExperts`; only the compute path differs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FusedGroupedExperts.Config):
        # No new fields in v1. derive() carries param_init / sharding_config /
        # dim / hidden_dim / num_experts from the converter's config by name;
        # any future knob must be re-declared here or derive() drops it.
        pass

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w13, DTensor):
            w13 = self.w13.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
        else:
            w13 = self.w13
            w2_EDF = self.w2_EDF

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        # The .bfloat16() casts stay OUTSIDE the Function so autograd handles
        # high-precision master-weight configs and dy reaches backward() BF16.
        y_RD = _MXFP8GroupedMLP.apply(
            x_RD.bfloat16(), w13.bfloat16(), w2_EDF.bfloat16(), offsets_E
        )
        return y_RD.type_as(x_RD)


def _decline(reason: str) -> None:
    logger.warning(f"mxfp8_grouped_experts override NOT applied: {reason}")


@override(
    target=RoutedExperts.Config,
    exact=True,
    description="Fused MXFP8 grouped-MLP kernels (A/B/C) for routed experts.",
)
def mxfp8_grouped_experts(cfg: RoutedExperts.Config) -> RoutedExperts.Config:
    """Swap converter-produced MXFP8 inner experts for the fused composite.

    Targets ``RoutedExperts.Config`` because it owns BOTH the
    ``token_dispatcher`` and ``inner_experts`` children — the only static
    place the ABI's ``m[g] % 128`` guarantee (dispatcher ``pad_multiple``) is
    checkable. Fires only when every gate passes; otherwise warns with the
    failed gate and returns ``cfg`` unchanged (fallback = the converter's
    unfused MXFP8 path).
    """
    experts = cfg.inner_experts
    dispatcher = cfg.token_dispatcher

    if not isinstance(dispatcher, TorchAOTokenDispatcher.Config):
        _decline(
            f"token_dispatcher is {type(dispatcher).__qualname__}, not a "
            "TorchAOTokenDispatcher.Config (no per-expert 128-row padding)."
        )
        return cfg
    if dispatcher.pad_multiple != _GROUP_ALIGNMENT:
        _decline(
            f"token_dispatcher.pad_multiple is {dispatcher.pad_multiple}; the "
            f"kernels require per-expert groups padded to {_GROUP_ALIGNMENT}."
        )
        return cfg
    if type(experts) is not _get_mxfp8_grouped_experts_cls(GroupedExperts).Config:
        _decline(
            f"inner_experts is {type(experts).__qualname__}, not the "
            "MXFP8GroupedExpertsConverter-produced MXFP8GroupedExperts.Config."
        )
        return cfg
    if experts.recipe_name != "mxfp8_rceil":
        _decline(
            f"inner_experts.recipe_name is '{experts.recipe_name}'; only "
            "'mxfp8_rceil' is supported."
        )
        return cfg
    if not _mxfp8_grouped_mlp_kernels_available:
        _decline(
            "torchao MXFP8 grouped-MLP kernels are unavailable in this "
            "environment (needs CUDA >= 12.8, SM 10.x, CuTe DSL runtime)."
        )
        return cfg
    if not (
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)
    ):
        _decline("requires CUDA device capability exactly (10, 0).")
        return cfg
    if not is_supported(
        experts.dim,
        experts.hidden_dim,
        _GROUP_ALIGNMENT,
        max(experts.num_experts, 1),
    ):
        _decline(
            f"is_supported(D={experts.dim}, F={experts.hidden_dim}) is False; "
            f"both dims must be positive multiples of {_GROUP_ALIGNMENT}."
        )
        return cfg

    fused = derive(experts, MXFP8FusedGroupedExperts.Config)
    # The w1_EFD/w3_EFD -> w13 param-init and sharding remaps are factory
    # work (they are not inherited through derive()).
    fused.param_init = _fuse_w13_grouped_experts_param_init(fused.param_init)
    if fused.sharding_config is not None:
        fused.sharding_config = _fuse_w13_grouped_experts_sharding(
            fused.sharding_config
        )
    cfg.inner_experts = fused
    return cfg
