# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Fused activation overrides for dense and grouped SwiGLU feed-forwards.

The default ``FeedForward`` already computes its gate and up projections with
one physical ``w13`` linear. ``fused_swiglu`` only replaces the torch-native
SiLU and multiply operations with a Triton kernel.

``dist_gemm_fused_swiglu`` preserves the dist-GEMM collective overlap while
using the same activation replacement. ``fused_grouped_experts`` fuses both
the grouped-expert projections and activation because the default
grouped-expert implementation still stores separate ``w1`` and ``w3``
parameters.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import spmd_types as spmd
import torch
import triton
import triton.language as tl

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from torchtitan.config import derive, override
from torchtitan.models.common.activation import ActivationFn, SwiGLU
from torchtitan.models.common.dist_gemm import DistGEMMFeedForward
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.sharding import ShardingConfig

__all__ = [
    "FusedGroupedExperts",
    "dist_gemm_fused_swiglu",
    "fused_grouped_experts",
    "silu_and_mul_backward_kernel",
    "silu_and_mul_forward_kernel",
    "silu_and_mul_op",
]


_MAX_BLOCK_N = 2048
_SILU_AND_MUL_BLOCK_M = 4


@triton.jit
def _silu_and_mul_forward_kernel(
    gate,
    up,
    out,
    offsets,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    NUM_OFFSETS: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    GATE_ROW_STRIDE: tl.constexpr,
    GATE_COL_STRIDE: tl.constexpr,
    UP_ROW_STRIDE: tl.constexpr,
    UP_COL_STRIDE: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    OUT_COL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Compute ``silu(gate) * up`` for optionally offset-limited rows."""
    row_start = tl.program_id(0) * BLOCK_M
    row_limit = NUM_ROWS
    if HAS_OFFSETS:
        row_limit = tl.load(offsets + NUM_OFFSETS - 1)
        if row_start >= row_limit:
            return

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < row_limit) & (cols[None, :] < NUM_COLS)

    gate_values = tl.load(
        gate + rows[:, None] * GATE_ROW_STRIDE + cols[None, :] * GATE_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up_values = tl.load(
        up + rows[:, None] * UP_ROW_STRIDE + cols[None, :] * UP_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    silu = gate_values * tl.sigmoid(gate_values)
    tl.store(
        out + rows[:, None] * OUT_ROW_STRIDE + cols[None, :] * OUT_COL_STRIDE,
        silu * up_values,
        mask=mask,
    )


@triton.jit
def _silu_and_mul_backward_kernel(
    grad_out,
    gate,
    up,
    grad_gate,
    grad_up,
    offsets,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    NUM_OFFSETS: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    GRAD_OUT_ROW_STRIDE: tl.constexpr,
    GRAD_OUT_COL_STRIDE: tl.constexpr,
    GATE_ROW_STRIDE: tl.constexpr,
    GATE_COL_STRIDE: tl.constexpr,
    UP_ROW_STRIDE: tl.constexpr,
    UP_COL_STRIDE: tl.constexpr,
    GRAD_GATE_ROW_STRIDE: tl.constexpr,
    GRAD_GATE_COL_STRIDE: tl.constexpr,
    GRAD_UP_ROW_STRIDE: tl.constexpr,
    GRAD_UP_COL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Backward for ``_silu_and_mul_forward_kernel`` over defined rows."""
    row_start = tl.program_id(0) * BLOCK_M
    row_limit = NUM_ROWS
    if HAS_OFFSETS:
        row_limit = tl.load(offsets + NUM_OFFSETS - 1)
        if row_start >= row_limit:
            return

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < row_limit) & (cols[None, :] < NUM_COLS)

    grad_values = tl.load(
        grad_out
        + rows[:, None] * GRAD_OUT_ROW_STRIDE
        + cols[None, :] * GRAD_OUT_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate_values = tl.load(
        gate + rows[:, None] * GATE_ROW_STRIDE + cols[None, :] * GATE_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up_values = tl.load(
        up + rows[:, None] * UP_ROW_STRIDE + cols[None, :] * UP_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sigmoid = tl.sigmoid(gate_values)
    silu = gate_values * sigmoid
    silu_grad = sigmoid * (1.0 + gate_values * (1.0 - sigmoid))

    tl.store(
        grad_gate
        + rows[:, None] * GRAD_GATE_ROW_STRIDE
        + cols[None, :] * GRAD_GATE_COL_STRIDE,
        grad_values * up_values * silu_grad,
        mask=mask,
    )
    tl.store(
        grad_up
        + rows[:, None] * GRAD_UP_ROW_STRIDE
        + cols[None, :] * GRAD_UP_COL_STRIDE,
        grad_values * silu,
        mask=mask,
    )


def silu_and_mul_forward_kernel(
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` with optional grouped_mm row offsets."""
    if offsets is not None and offsets.numel() == 0:
        raise ValueError("offsets must be non-empty when provided.")
    out = torch.empty_like(gate, memory_format=torch.contiguous_format)

    block_m = _SILU_AND_MUL_BLOCK_M
    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(gate.shape[1]))
    grid = (triton.cdiv(gate.shape[0], block_m), triton.cdiv(gate.shape[1], block_n))
    _silu_and_mul_forward_kernel[grid](
        gate,
        up,
        out,
        offsets if offsets is not None else gate,
        NUM_ROWS=gate.shape[0],
        NUM_COLS=gate.shape[1],
        NUM_OFFSETS=offsets.numel() if offsets is not None else 0,
        HAS_OFFSETS=offsets is not None,
        GATE_ROW_STRIDE=gate.stride(0),
        GATE_COL_STRIDE=gate.stride(1),
        UP_ROW_STRIDE=up.stride(0),
        UP_COL_STRIDE=up.stride(1),
        OUT_ROW_STRIDE=out.stride(0),
        OUT_COL_STRIDE=out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out


def silu_and_mul_backward_kernel(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if offsets is not None and offsets.numel() == 0:
        raise ValueError("offsets must be non-empty when provided.")
    grad_gate = torch.empty_like(gate, memory_format=torch.contiguous_format)
    grad_up = torch.empty_like(up, memory_format=torch.contiguous_format)

    block_m = _SILU_AND_MUL_BLOCK_M
    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(gate.shape[1]))
    grid = (triton.cdiv(gate.shape[0], block_m), triton.cdiv(gate.shape[1], block_n))
    _silu_and_mul_backward_kernel[grid](
        grad_out,
        gate,
        up,
        grad_gate,
        grad_up,
        offsets if offsets is not None else gate,
        NUM_ROWS=gate.shape[0],
        NUM_COLS=gate.shape[1],
        NUM_OFFSETS=offsets.numel() if offsets is not None else 0,
        HAS_OFFSETS=offsets is not None,
        GRAD_OUT_ROW_STRIDE=grad_out.stride(0),
        GRAD_OUT_COL_STRIDE=grad_out.stride(1),
        GATE_ROW_STRIDE=gate.stride(0),
        GATE_COL_STRIDE=gate.stride(1),
        UP_ROW_STRIDE=up.stride(0),
        UP_COL_STRIDE=up.stride(1),
        GRAD_GATE_ROW_STRIDE=grad_gate.stride(0),
        GRAD_GATE_COL_STRIDE=grad_gate.stride(1),
        GRAD_UP_ROW_STRIDE=grad_up.stride(0),
        GRAD_UP_COL_STRIDE=grad_up.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return grad_gate, grad_up


@torch.library.custom_op(
    "torchtitan::silu_and_mul",
    mutates_args=(),
    device_types="cuda",
)
def silu_and_mul_op(
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` over optionally offset-limited rows."""
    return silu_and_mul_forward_kernel(gate, up, offsets)


@silu_and_mul_op.register_fake
def silu_and_mul_op_fake(
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(gate, memory_format=torch.contiguous_format)


@torch.library.custom_op(
    "torchtitan::silu_and_mul_backward",
    mutates_args=(),
    device_types="cuda",
)
def silu_and_mul_backward_op(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for ``torchtitan::silu_and_mul``."""
    return silu_and_mul_backward_kernel(grad_out, gate, up, offsets)


@silu_and_mul_backward_op.register_fake
def silu_and_mul_backward_op_fake(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(gate, memory_format=torch.contiguous_format),
        torch.empty_like(up, memory_format=torch.contiguous_format),
    )


def silu_and_mul_autograd_backward(ctx, grad_out):
    if ctx.has_offsets:
        gate, up, offsets = ctx.saved_tensors
    else:
        gate, up = ctx.saved_tensors
        offsets = None
    grad_gate, grad_up = silu_and_mul_backward_op(
        grad_out,
        gate,
        up,
        offsets,
    )
    return grad_gate, grad_up, None


def silu_and_mul_setup_context(ctx, inputs, output):
    gate, up = inputs[:2]
    offsets = inputs[2] if len(inputs) > 2 else None
    ctx.has_offsets = offsets is not None
    if offsets is None:
        ctx.save_for_backward(gate, up)
    else:
        ctx.save_for_backward(gate, up, offsets)


silu_and_mul_op.register_autograd(
    silu_and_mul_autograd_backward, setup_context=silu_and_mul_setup_context
)


def _make_fused_gate_up_init(
    gate_init: Callable,
    up_init: Callable,
    *,
    gate_up_axis: int,
) -> Callable:
    """Build an initializer for a fused gate/up weight from per-half initializers.

    The fused weight has a size-2 ``gate_up_axis`` (index 0 = gate / stock w1,
    index 1 = up / stock w3). Each half is initialized with its own initializer
    because the gate and up projections differ (e.g. up shares w2's depth-scaled
    init), so initializing the whole tensor at once would mis-init the up half.
    Used by the grouped FusedGroupedExperts ``(E, F, 2, D)`` override and by
    the logical 3D view of the dense fused linear weight.
    """

    def _init(t: torch.Tensor) -> None:
        gate_idx: list[int | slice] = [slice(None)] * t.ndim
        up_idx: list[int | slice] = [slice(None)] * t.ndim
        gate_idx[gate_up_axis] = 0
        up_idx[gate_up_axis] = 1
        gate_init(t[tuple(gate_idx)])  # gate (stock w1)
        up_init(t[tuple(up_idx)])  # up (stock w3)

    return _init


def _fused_silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` via the fused ``torchtitan::silu_and_mul`` op."""
    if isinstance(gate, DTensor):
        assert isinstance(up, DTensor)
        placements = gate.placements
        mapped = local_map(
            _silu_and_mul_2d,
            out_placements=(placements,),
            in_placements=(placements, placements),
            in_grad_placements=(placements, placements),
            device_mesh=gate.device_mesh,
        )
        return mapped(gate, up)
    return _silu_and_mul_2d(gate, up)


def _silu_and_mul_2d(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # TODO(pianpwk): Migrate this local_map workaround to a custom op SPMD
    # propagation rule registration system.
    activation_type = (
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
        spmd.PartitionSpec(("dp", "cp"), "tp"),
    )
    return spmd.local_map(
        in_types=(activation_type, activation_type),
        out_types=activation_type,
    )(
        lambda gate, up: silu_and_mul_op(
            gate.reshape(-1, gate.shape[-1]),
            up.reshape(-1, up.shape[-1]),
        ).reshape(gate.shape)
    )(
        gate, up
    )


def _replace_swiglu_activation(cfg: FeedForward.Config) -> FeedForward.Config:
    """Replace the torch-native SwiGLU callable with the fused implementation."""
    if not isinstance(cfg.activation_fn.fn, SwiGLU):
        raise ValueError(
            "The fused_swiglu override requires the default SwiGLU activation, "
            f"but found {type(cfg.activation_fn.fn).__name__}."
        )
    return replace(
        cfg,
        activation_fn=ActivationFn.Config(fn=_fused_silu_and_mul),
    )


@override(
    target=FeedForward.Config,
    exact=True,
    description="Fuse the SwiGLU SiLU and multiply operations with Triton.",
)
def fused_swiglu(cfg: FeedForward.Config) -> FeedForward.Config:
    return _replace_swiglu_activation(cfg)


@override(
    target=DistGEMMFeedForward.Config,
    exact=True,
    description="Fuse SwiGLU activation while preserving dist-GEMM TP overlap.",
)
def dist_gemm_fused_swiglu(
    cfg: DistGEMMFeedForward.Config,
) -> DistGEMMFeedForward.Config:
    return _replace_swiglu_activation(cfg)


class FusedGroupedExperts(GroupedExperts):
    """Routed experts (grouped GEMM) with the gate and up projections fused.

    ``w13`` has shape ``(num_experts, hidden_dim, 2, dim)``: ``w13[:, :, 0]`` is
    the gate (original ``w1_EFD``) and ``w13[:, :, 1]`` the up (original ``w3_EFD``). A
    single grouped GEMM computes both projections; the fused
    ``torchtitan::silu_and_mul`` op forms the activation (skipping inactive
    capacity-padding rows via grouped_mm offsets). The down projection
    ``w2_EDF`` is reused as-is.

    The explicit ``2`` axis stays unsharded and matches the logical
    ``(hidden_dim, 2, dim)`` view used by the dense fused ``w13``. TP shards
    ``hidden_dim`` (dim 1) and EP shards the expert axis (dim 0), so each rank
    keeps matching gate/up slices. Checkpoints save original ``w1_EFD`` /
    ``w3_EFD`` separately.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

        # delete separate w1/w3 and fuse
        del self.w1_EFD
        del self.w3_EFD
        self.w13 = torch.nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, 2, config.dim)
        )

        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

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

        E, F, _, D = w13.shape
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)

        # The fused parameter stores gate and up interleaved as (E, F, 2, D);
        # the grouped GEMM consumes them as one (E, 2F, D) expert weight.
        w13_E_2F_D = w13.bfloat16().reshape(E, F * 2, D)
        gate_up_R2F = self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=w13_E_2F_D, offs=offsets_E
        )
        gate_RF, up_RF = gate_up_R2F.reshape(-1, F, 2).unbind(-1)
        h_RF = silu_and_mul_op(gate_RF, up_RF, offsets_E)
        return self._grouped_mm(A=h_RF, weight_EOI=w2_EDF, offs=offsets_E).type_as(x_RD)

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Save fused as ``w1_EFD`` / ``w3_EFD`` on save."""
        w13 = state_dict.pop(f"{prefix}w13")
        state_dict[f"{prefix}w1_EFD"] = w13[:, :, 0, :].contiguous()
        state_dict[f"{prefix}w3_EFD"] = w13[:, :, 1, :].contiguous()

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        """Combine separate ``w1_EFD`` / ``w3_EFD`` back into the fused ``w13`` on load."""
        w1_key, w3_key = f"{prefix}w1_EFD", f"{prefix}w3_EFD"
        if w1_key in state_dict and w3_key in state_dict:
            state_dict[f"{prefix}w13"] = torch.stack(
                [state_dict.pop(w1_key), state_dict.pop(w3_key)], dim=2
            )


def _fuse_w13_grouped_experts_param_init(param_init: dict | None) -> dict | None:
    """Remap ``w1_EFD`` / ``w3_EFD`` initializers onto the fused ``w13``.

    Other entries (e.g. ``w2_EDF``) are kept as-is.
    """
    if param_init is None:
        return None
    w1_init = param_init.get("w1_EFD")
    w3_init = param_init.get("w3_EFD")
    fused = {k: v for k, v in param_init.items() if k not in ("w1_EFD", "w3_EFD")}
    if w1_init is not None and w3_init is not None:
        fused["w13"] = _make_fused_gate_up_init(w1_init, w3_init, gate_up_axis=2)
    return fused or None


def _fuse_w13_grouped_experts_sharding(base: ShardingConfig) -> ShardingConfig:
    """Replace the ``w1_EFD`` / ``w3_EFD`` state shardings with one for ``w13``.

    ``w13`` (E, F, 2, D) shards on the same axes as ``w1_EFD`` (E, F, D): EP on
    dim 0 (expert) and TP on dim 1 (hidden); the ``2`` axis (dim 2) stays
    unsharded. Everything else (``w2_EDF``, local_map, in/out shardings) is kept.
    """
    state = dict(base.state_shardings)
    w1_layout = state.pop("w1_EFD")
    state.pop("w3_EFD")
    state["w13"] = w1_layout
    return replace(base, state_shardings=state)


@override(
    target=GroupedExperts.Config,
    description="Fuse routed-experts gate+up into one weight; fused SiLU-and-mul.",
)
def fused_grouped_experts(
    cfg: GroupedExperts.Config,
) -> GroupedExperts.Config:
    # Remap w1_EFD/w3_EFD param-init and state shardings onto the fused w13.
    # Idempotent: return cfg unchanged if it is not a stock GroupedExperts.Config
    # (already fused, or a subclass like GptOssGroupedExperts).
    if type(cfg) is not GroupedExperts.Config:
        return cfg

    param_init = _fuse_w13_grouped_experts_param_init(cfg.param_init)
    fused = derive(cfg, FusedGroupedExperts.Config, param_init=param_init)
    base = cfg.sharding_config
    if base is not None:
        fused.sharding_config = _fuse_w13_grouped_experts_sharding(base)
    return fused
