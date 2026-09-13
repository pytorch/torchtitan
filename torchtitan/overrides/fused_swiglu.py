# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Fused SwiGLU overrides.

``fused_swiglu`` replaces every ``SwiGLU`` activation selected by the override
framework. ``fused_grouped_experts`` retains the grouped gate/up projection
override until that projection becomes the core ``GroupedExperts`` default.
"""

from dataclasses import dataclass, replace
from typing import Any

import spmd_types as spmd
import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.models.common.activation import ActivationFn, SwiGLU
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.sharding import ShardingConfig

__all__ = [
    "FusedGroupedExperts",
    "FusedSwiGLU",
    "fused_grouped_experts",
    "fused_swiglu",
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


class FusedSwiGLU(ActivationFn):
    """SwiGLU activation implemented by the fused Triton operation."""

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
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


@override(
    target=SwiGLU.Config,
    exact=True,
    description="Fuse the SwiGLU SiLU and multiply operations with Triton.",
)
def fused_swiglu(cfg: SwiGLU.Config) -> FusedSwiGLU.Config:
    return derive(cfg, FusedSwiGLU.Config)


class FusedGroupedExperts(GroupedExperts):
    """Grouped experts with one physical interleaved gate/up parameter."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
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
        E, F, _, D = self.w13.shape
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        gate_up_R2F = self._grouped_mm(
            A=x_RD.bfloat16(),
            weight_EOI=self.w13.reshape(E, F * 2, D),
            offs=offsets_E,
        )
        gate_RF, up_RF = gate_up_R2F.reshape(-1, F, 2).unbind(-1)
        h_RF = silu_and_mul_op(gate_RF, up_RF, offsets_E)
        return self._grouped_mm(A=h_RF, weight_EOI=self.w2_EDF, offs=offsets_E).type_as(
            x_RD
        )

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Expose fused experts under the logical w1/w3 checkpoint keys."""
        w13 = state_dict.pop(f"{prefix}w13")
        state_dict[f"{prefix}w1_EFD"] = w13[:, :, 0, :].contiguous()
        state_dict[f"{prefix}w3_EFD"] = w13[:, :, 1, :].contiguous()

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        """Pack logical w1/w3 checkpoint entries into the fused parameter."""
        gate_key = f"{prefix}w1_EFD"
        up_key = f"{prefix}w3_EFD"
        if gate_key not in state_dict or up_key not in state_dict:
            return
        state_dict[f"{prefix}w13"] = torch.stack(
            [state_dict.pop(gate_key), state_dict.pop(up_key)], dim=2
        )


def _fuse_grouped_experts_param_init(param_init: dict | None) -> dict | None:
    """Remap logical gate/up initializers onto the fused parameter."""
    if param_init is None:
        return None
    gate_init = param_init.get("w1_EFD")
    up_init = param_init.get("w3_EFD")
    fused = {
        key: value
        for key, value in param_init.items()
        if key not in ("w1_EFD", "w3_EFD")
    }
    if gate_init is not None and up_init is not None:

        def init_w13(w13: torch.Tensor) -> None:
            gate_init(w13[:, :, 0, :])
            up_init(w13[:, :, 1, :])

        fused["w13"] = init_w13
    return fused or None


def _fuse_grouped_experts_sharding(base: ShardingConfig) -> ShardingConfig:
    """Replace logical gate/up shardings with the fused parameter sharding."""
    state = dict(base.state_shardings)
    gate_layout = state.pop("w1_EFD")
    state.pop("w3_EFD")
    state["w13"] = gate_layout
    return replace(base, state_shardings=state)


@override(
    target=GroupedExperts.Config,
    description="Fuse routed-experts gate/up projection and SwiGLU activation.",
)
def fused_grouped_experts(cfg: GroupedExperts.Config) -> GroupedExperts.Config:
    if type(cfg) is not GroupedExperts.Config:
        return cfg

    fused = derive(
        cfg,
        FusedGroupedExperts.Config,
        param_init=_fuse_grouped_experts_param_init(cfg.param_init),
    )
    if cfg.sharding_config is not None:
        fused.sharding_config = _fuse_grouped_experts_sharding(cfg.sharding_config)
    return fused
