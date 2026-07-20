# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""
Example override: a SwiGLU feed-forward with a single fused gate+up weight.

This is the worked example referenced in ``torchtitan/overrides/README.md``. It
demonstrates the pieces a non-trivial fused module needs to plug in via the
override mechanism, *without touching core*:

1. Custom ``__init__`` parametrization — ``w1`` and ``w3`` of the stock
   :class:`FeedForward` are fused into one parameter ``w13`` of shape
   ``(hidden_dim, 2, dim)`` (``w13[:, 0]`` is the gate / stock ``w1``,
   ``w13[:, 1]`` the up / stock ``w3``). The gate+up projection is a single GEMM.
2. Weight initialization via the ``Module`` protocol (``param_init``).
3. A ``sharding_config`` that supports both FSDP and tensor parallelism.
4. Registration via ``@override`` targeting ``FeedForward.Config``.

This module also defines the ``torchtitan::silu_and_mul`` custom CUDA op (a fused
SiLU-and-mul Triton kernel) and :class:`FusedGroupedExperts`, which applies the
same gate+up fusion to MoE experts. Both the ``fused_swiglu`` (``FeedForward``)
and ``fused_grouped_experts`` (``GroupedExperts``) overrides are registered here;
activate each by naming its factory, e.g. ``--override.imports
torchtitan.overrides.fused_swiglu.fused_swiglu,torchtitan.overrides.fused_swiglu.fused_grouped_experts``.
For the DeepEP inference path, pair it with the sibling ``deepep_override`` dispatcher
override (``torchtitan.overrides.moe_dispatch_override``).

Tensor parallelism — the ``(hidden_dim, 2, dim)`` layout is what makes TP work.
``w13`` is sharded ``Shard(0)`` on the ``hidden_dim`` axis, so each TP rank holds
``(hidden_dim/tp, 2, dim)`` — a matching slice of *both* the gate and up
projections (the Megatron column-parallel layout for gated MLPs). A flat
``(2*hidden, dim)`` weight has no correct TP sharding (``Shard(0)`` there would
give one rank all of ``w1`` and another all of ``w3``); the explicit ``2`` dim
fixes that. ``hidden_dim`` is also dim 0, so FSDP shards the large axis cleanly
at any degree. The layout is contiguous and transpose-free, so it costs nothing
at compute time: the single GEMM is expressed with ``einsum`` (which contracts
``dim`` and keeps ``hidden`` sharded), and never reshapes across the sharded axis.

NOTE (checkpoint compatibility) -- although the parameter is the fused ``w13``,
this module checkpoints in the stock ``FeedForward`` layout
(``w1.weight`` / ``w3.weight``), so its checkpoints **are** interchangeable with
the non-fused module (and with the HF adapter, which targets the stock layout). A
``register_state_dict_post_hook`` splits ``w13`` into ``w1.weight``/``w3.weight``
on save, and a ``register_load_state_dict_pre_hook`` merges them back into ``w13``
on load (a native ``w13`` key is still accepted for back-compat). See
``torchtitan/overrides/README.md`` "Checkpoint Compatibility".
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
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.sharding import ShardingConfig

__all__ = [
    "FusedGroupedExperts",
    "FusedSwiGLU",
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
    Shared by the dense FusedSwiGLU ``(hidden, 2, dim)`` (axis 1) and the grouped
    FusedGroupedExperts ``(E, F, 2, D)`` (axis 2) overrides.
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
    return spmd.local_map(
        in_types=(
            {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},  # gate_BLF
            {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},  # up_BLF
        ),
        out_types={"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
    )(
        lambda gate, up: silu_and_mul_op(
            gate.reshape(-1, gate.shape[-1]),
            up.reshape(-1, up.shape[-1]),
        ).reshape(gate.shape)
    )(
        gate, up
    )


class FusedSwiGLU(FeedForward):
    """SwiGLU FFN with the gate and up projections fused into one parameter.

    ``w13`` has shape ``(hidden_dim, 2, dim)``: ``w13[:, 0]`` is the gate
    projection (the stock ``w1``) and ``w13[:, 1]`` the up projection (the stock
    ``w3``). A single GEMM computes both; the result is split into gate/up. The
    down projection ``w2`` is reused as-is. ``hidden_dim`` is dim 0, so TP shards
    it (``Shard(0)``, matching gate/up slices per rank) and FSDP shards it cleanly
    at any degree. Inherits :class:`FeedForward` (building then deleting ``w1`` /
    ``w3``) so ``isinstance(x, FeedForward)`` checks still hold.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        del self.w1
        del self.w3
        self.w13 = torch.nn.Parameter(
            torch.empty(config.w1.out_features, 2, config.w1.in_features)
        )
        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = torch.einsum("...d,hgd->...hg", x, self.w13).unbind(-1)
        return self.w2(_fused_silu_and_mul(gate, up))

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Emit the fused ``w13`` as the stock ``w1.weight`` / ``w3.weight``.

        Runs after the default ``state_dict`` (which produced ``{prefix}w13``),
        so the saved checkpoint is in the stock FeedForward layout. ``w13[:, 0]``
        is the gate (stock ``w1``) and ``w13[:, 1]`` the up (stock ``w3``).
        """
        w13 = state_dict.pop(f"{prefix}w13")
        state_dict[f"{prefix}w1.weight"] = w13[:, 0].contiguous()
        state_dict[f"{prefix}w3.weight"] = w13[:, 1].contiguous()

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        """Merge stock ``w1.weight`` / ``w3.weight`` back into the fused ``w13``.

        Runs before the default ``_load_from_state_dict``, so it loads ``w13``
        (the real parameter) normally afterwards. A native ``w13`` key is left
        untouched, keeping back-compat with checkpoints saved as ``w13``.
        """
        w1_key, w3_key = f"{prefix}w1.weight", f"{prefix}w3.weight"
        if w1_key in state_dict and w3_key in state_dict:
            state_dict[f"{prefix}w13"] = torch.stack(
                [state_dict.pop(w1_key), state_dict.pop(w3_key)], dim=1
            )


@override(
    "fused_swiglu",
    target=FeedForward.Config,
    description="Fuse SwiGLU gate+up into one weight (FSDP + TP).",
)
def fused_swiglu(cfg: FeedForward.Config) -> FusedSwiGLU.Config:
    w1_init = (cfg.w1.param_init or {}).get("weight")
    w3_init = (cfg.w3.param_init or {}).get("weight")
    param_init = None
    if w1_init is not None and w3_init is not None:
        param_init = {"w13": _make_fused_gate_up_init(w1_init, w3_init, gate_up_axis=1)}

    fused = derive(cfg, FusedSwiGLU.Config, param_init=param_init)

    base = cfg.sharding_config
    fused.sharding_config = ShardingConfig(
        state_shardings={"w13": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings=base.in_src_shardings if base is not None else None,
        in_dst_shardings=base.in_dst_shardings if base is not None else None,
    )
    return fused


class FusedGroupedExperts(GroupedExperts):
    """Routed experts (grouped GEMM) with the gate and up projections fused.

    ``w13`` has shape ``(num_experts, hidden_dim, 2, dim)``: ``w13[:, :, 0]`` is
    the gate (original ``w1_EFD``) and ``w13[:, :, 1]`` the up (original ``w3_EFD``). A
    single grouped GEMM computes both projections; the fused
    ``torchtitan::silu_and_mul`` op forms the activation (skipping inactive
    capacity-padding rows via grouped_mm offsets). The down projection
    ``w2_EDF`` is reused as-is.

    The explicit ``2`` axis stays unsharded so that it matches the dense FusedSwiGLU
    ``(hidden_dim, 2, dim)`` layout: TP shards ``hidden_dim`` (dim 1) and EP
    shards the expert axis (dim 0), so each rank keeps matching gate/up slices.
    Checkpoints save original ``w1_EFD`` / ``w3_EFD`` separately.
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

        w13_E_D_2F = w13.bfloat16().reshape(E, F * 2, D).transpose(-2, -1)
        gate_up_R2F = torch._grouped_mm(x_RD.bfloat16(), w13_E_D_2F, offs=offsets_E)
        gate_RF, up_RF = gate_up_R2F.reshape(-1, F, 2).unbind(-1)
        h_RF = silu_and_mul_op(gate_RF, up_RF, offsets_E)
        return torch._grouped_mm(
            h_RF, w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E
        ).type_as(x_RD)

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
    "fused_grouped_experts",
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
