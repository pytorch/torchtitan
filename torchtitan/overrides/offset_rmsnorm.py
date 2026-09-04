# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused Triton OffsetRMSNorm override for Qwen3.5.

The stock module computes ``(1 + weight) * rmsnorm(input)`` with eager
PyTorch operations. This override keeps that parameterization and performs the
normalization and offset scaling in one Triton kernel. The backward uses Triton
kernels for both the input and weight gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl
from spmd_types import SpmdType
from torch.distributed.tensor import DTensor

from torchtitan.config import derive, override
from torchtitan.models.qwen3_5.model import OffsetRMSNorm
from torchtitan.protocols.sharding import LocalMapConfig, resolve_placements


__all__ = [
    "TritonOffsetRMSNorm",
    "triton_offset_rms_norm",
    "triton_offset_rmsnorm",
]


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_BLOCK_SIZE = 65536
_DW_BLOCK_M = 32
_DW_BLOCK_N = 64


def _num_warps(block_size: int) -> int:
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


@triton.jit
def _offset_rms_norm_forward_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    inverse_rms_ptr,
    num_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    input_row = input_ptr + row_idx * num_cols
    output_row = output_ptr + row_idx * num_cols
    input_fp32 = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(input_fp32 * input_fp32, axis=0) / num_cols
    inverse_rms = tl.rsqrt(variance + eps)
    weight_fp32 = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    output_fp32 = input_fp32 * inverse_rms * (1.0 + weight_fp32)

    tl.store(output_row + col_offsets, output_fp32, mask=mask)
    tl.store(inverse_rms_ptr + row_idx, inverse_rms)


@triton.jit
def _offset_rms_norm_input_grad_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    inverse_rms_ptr,
    grad_input_ptr,
    num_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols
    row_offsets = row_idx * num_cols + col_offsets

    grad_output_fp32 = tl.load(grad_output_ptr + row_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    input_fp32 = tl.load(input_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    weight_fp32 = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.load(inverse_rms_ptr + row_idx)

    scaled_grad = grad_output_fp32 * (1.0 + weight_fp32)
    projection = tl.sum(scaled_grad * input_fp32, axis=0)
    grad_input_fp32 = inverse_rms * scaled_grad
    grad_input_fp32 -= (
        input_fp32 * inverse_rms * inverse_rms * inverse_rms * projection / num_cols
    )
    tl.store(grad_input_ptr + row_offsets, grad_input_fp32, mask=mask)


@triton.jit
def _offset_rms_norm_weight_grad_partial_kernel(
    grad_output_ptr,
    input_ptr,
    inverse_rms_ptr,
    partial_grad_weight_ptr,
    num_rows,
    num_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    col_block_idx = tl.program_id(0)
    row_block_idx = tl.program_id(1)
    row_offsets = row_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = row_offsets[:, None] * num_cols + col_offsets[None, :]
    mask = (row_offsets[:, None] < num_rows) & (col_offsets[None, :] < num_cols)

    grad_output_fp32 = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    input_fp32 = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.load(
        inverse_rms_ptr + row_offsets,
        mask=row_offsets < num_rows,
        other=0.0,
    )
    partial = tl.sum(
        grad_output_fp32 * input_fp32 * inverse_rms[:, None],
        axis=0,
    )
    partial_offsets = row_block_idx * num_cols + col_offsets
    tl.store(
        partial_grad_weight_ptr + partial_offsets,
        partial,
        mask=col_offsets < num_cols,
    )


@triton.jit
def _offset_rms_norm_weight_grad_reduce_kernel(
    partial_grad_weight_ptr,
    grad_weight_ptr,
    num_partials,
    num_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    col_block_idx = tl.program_id(0)
    partial_offsets = tl.arange(0, BLOCK_M)
    col_offsets = col_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = partial_offsets[:, None] * num_cols + col_offsets[None, :]
    mask = (partial_offsets[:, None] < num_partials) & (col_offsets[None, :] < num_cols)
    partial = tl.load(
        partial_grad_weight_ptr + offsets,
        mask=mask,
        other=0.0,
    )
    grad_weight = tl.sum(partial, axis=0)
    tl.store(
        grad_weight_ptr + col_offsets,
        grad_weight,
        mask=col_offsets < num_cols,
    )


@torch.library.triton_op("torchtitan::triton_offset_rms_norm", mutates_args={})
def _triton_offset_rms_norm_op(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    input = input.contiguous()
    weight = weight.contiguous()
    num_cols = input.shape[-1]
    block_size = triton.next_power_of_2(num_cols)
    if block_size > _MAX_BLOCK_SIZE:
        raise ValueError(
            f"Triton OffsetRMSNorm supports at most {_MAX_BLOCK_SIZE} columns, "
            f"got {num_cols}"
        )
    num_rows = input.numel() // num_cols
    output = torch.empty_like(input)
    inverse_rms = torch.empty(num_rows, dtype=torch.float32, device=input.device)
    torch.library.wrap_triton(_offset_rms_norm_forward_kernel)[(num_rows,)](
        input,
        weight,
        output,
        inverse_rms,
        num_cols=num_cols,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return output, inverse_rms


@torch.library.triton_op("torchtitan::triton_offset_rms_norm_backward", mutates_args={})
def _triton_offset_rms_norm_backward_op(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    inverse_rms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_output = grad_output.contiguous()
    num_cols = input.shape[-1]
    num_rows = input.numel() // num_cols
    block_size = triton.next_power_of_2(num_cols)

    grad_input = torch.empty_like(input)
    torch.library.wrap_triton(_offset_rms_norm_input_grad_kernel)[(num_rows,)](
        grad_output,
        input,
        weight,
        inverse_rms,
        grad_input,
        num_cols=num_cols,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )

    num_partials = triton.cdiv(num_rows, _DW_BLOCK_M)
    partial_grad_weight = torch.empty(
        num_partials,
        num_cols,
        dtype=torch.float32,
        device=input.device,
    )
    num_col_blocks = triton.cdiv(num_cols, _DW_BLOCK_N)
    torch.library.wrap_triton(_offset_rms_norm_weight_grad_partial_kernel)[
        (num_col_blocks, num_partials)
    ](
        grad_output,
        input,
        inverse_rms,
        partial_grad_weight,
        num_rows,
        num_cols=num_cols,
        BLOCK_M=_DW_BLOCK_M,
        BLOCK_N=_DW_BLOCK_N,
        num_warps=4,
    )

    grad_weight = torch.empty_like(weight)
    reduce_block_m = triton.next_power_of_2(num_partials)
    torch.library.wrap_triton(_offset_rms_norm_weight_grad_reduce_kernel)[
        (num_col_blocks,)
    ](
        partial_grad_weight,
        grad_weight,
        num_partials,
        num_cols=num_cols,
        BLOCK_M=reduce_block_m,
        BLOCK_N=_DW_BLOCK_N,
        num_warps=_num_warps(reduce_block_m),
    )
    return grad_input, grad_weight


def _triton_offset_rms_norm_setup_context(ctx, inputs, output) -> None:
    input, weight, _eps = inputs
    _output, inverse_rms = output
    ctx.save_for_backward(input, weight, inverse_rms)


def _triton_offset_rms_norm_autograd_backward(
    ctx,
    grad_output: torch.Tensor,
    _grad_inverse_rms: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    input, weight, inverse_rms = ctx.saved_tensors
    grad_input, grad_weight = _triton_offset_rms_norm_backward_op(
        grad_output.contiguous(),
        input,
        weight,
        inverse_rms,
    )
    return grad_input, grad_weight, None


_triton_offset_rms_norm_op.register_autograd(
    _triton_offset_rms_norm_autograd_backward,
    setup_context=_triton_offset_rms_norm_setup_context,
)


def triton_offset_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute ``(1 + weight) * rmsnorm(input)`` with Triton."""
    output, _inverse_rms = _triton_offset_rms_norm_op(
        input.contiguous(),
        weight.contiguous(),
        eps,
    )
    return output


class TritonOffsetRMSNorm(OffsetRMSNorm):
    """Qwen3.5 OffsetRMSNorm implemented by a fused Triton kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(OffsetRMSNorm.Config):
        weight_grad_sharding: SpmdType | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.weight_grad_sharding = config.weight_grad_sharding

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self, input: torch.Tensor
    ) -> torch.Tensor:
        if not input.is_cuda or input.dtype not in _SUPPORTED_DTYPES:
            return super().forward(input)

        weight = self.weight
        if isinstance(weight, DTensor):
            if self.weight_grad_sharding is None:
                raise AssertionError(
                    "DTensor weight requires a configured gradient sharding"
                )
            weight = weight.to_local(
                grad_placements=resolve_placements(
                    self.weight_grad_sharding,
                    weight.device_mesh,
                )
            )
        return triton_offset_rms_norm(input, weight, self.eps)


@override(
    target=OffsetRMSNorm.Config,
    exact=True,
    description="Use a fused Triton forward/backward kernel for Qwen3.5 OffsetRMSNorm.",
)
def triton_offset_rmsnorm(
    cfg: OffsetRMSNorm.Config,
) -> TritonOffsetRMSNorm.Config:
    sharding_config = cfg.sharding_config
    weight_grad_sharding = None
    if sharding_config is not None:
        input_shardings = (
            sharding_config.in_dst_shardings or sharding_config.in_src_shardings or {}
        )
        input_sharding = input_shardings.get("input")
        output_sharding = (
            sharding_config.out_src_shardings or sharding_config.out_dst_shardings
        )
        weight_sharding = sharding_config.state_shardings.get("weight")
        if input_sharding is None or output_sharding is None:
            raise ValueError(
                "Triton OffsetRMSNorm requires input and output sharding "
                "contracts when a sharding config is present"
            )
        if weight_sharding is None:
            raise ValueError("Triton OffsetRMSNorm requires a weight sharding contract")
        weight_grad_sharding = SpmdType(
            {
                axis: axis_type.backward_type()
                for axis, axis_type in weight_sharding.local_type.items()
            },
            partition_spec=weight_sharding.partition_spec,
        )
        sharding_config = replace(
            sharding_config,
            local_map=LocalMapConfig(in_grad_placements=(input_sharding,)),
        )
    return derive(
        cfg,
        TritonOffsetRMSNorm.Config,
        sharding_config=sharding_config,
        weight_grad_sharding=weight_grad_sharding,
    )
