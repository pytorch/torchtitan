# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused Triton RMSNorm-gate override for Kimi K3.

For each token and attention head, the stock module casts the input, weight,
and gate to FP32 and computes
``input * rsqrt(mean(input**2, dim=-1) + eps) * weight * sigmoid(gate)``.
The result is then cast back to the input dtype. This override fuses those
operations in one forward kernel. The backward uses Triton kernels for the
input, gate, and weight gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.models.common.decoder_sharding import attention_activation_placement
from torchtitan.models.kimi_k3.kda import KimiRMSNormGated


__all__ = [
    "TritonKimiRMSNormGated",
    "triton_kimi_rms_norm_gated",
    "triton_kimi_rmsnorm_gated",
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
def _kimi_rms_norm_gated_forward_kernel(
    input_ptr,
    gate_ptr,
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
    row_offsets = row_idx * num_cols + col_offsets

    input_fp32 = tl.load(input_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(input_fp32 * input_fp32, axis=0) / num_cols
    inverse_rms = tl.rsqrt(variance + eps)
    weight_fp32 = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    gate_fp32 = tl.load(gate_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    output_fp32 = input_fp32 * inverse_rms * weight_fp32 * tl.sigmoid(gate_fp32)

    tl.store(output_ptr + row_offsets, output_fp32, mask=mask)
    tl.store(inverse_rms_ptr + row_idx, inverse_rms)


@triton.jit
def _kimi_rms_norm_gated_input_gate_grad_kernel(
    grad_output_ptr,
    input_ptr,
    gate_ptr,
    weight_ptr,
    inverse_rms_ptr,
    grad_input_ptr,
    grad_gate_ptr,
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
    gate_fp32 = tl.load(gate_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    weight_fp32 = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.load(inverse_rms_ptr + row_idx)
    sigmoid_gate = tl.sigmoid(gate_fp32)

    scaled_grad = grad_output_fp32 * weight_fp32 * sigmoid_gate
    projection = tl.sum(scaled_grad * input_fp32, axis=0)
    grad_input_fp32 = inverse_rms * scaled_grad
    grad_input_fp32 -= (
        input_fp32 * inverse_rms * inverse_rms * inverse_rms * projection / num_cols
    )
    grad_gate_fp32 = (
        grad_output_fp32
        * input_fp32
        * inverse_rms
        * weight_fp32
        * sigmoid_gate
        * (1.0 - sigmoid_gate)
    )

    tl.store(grad_input_ptr + row_offsets, grad_input_fp32, mask=mask)
    tl.store(grad_gate_ptr + row_offsets, grad_gate_fp32, mask=mask)


@triton.jit
def _kimi_rms_norm_gated_weight_grad_partial_kernel(
    grad_output_ptr,
    input_ptr,
    gate_ptr,
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
    gate_fp32 = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.load(
        inverse_rms_ptr + row_offsets,
        mask=row_offsets < num_rows,
        other=0.0,
    )
    partial = tl.sum(
        grad_output_fp32 * input_fp32 * inverse_rms[:, None] * tl.sigmoid(gate_fp32),
        axis=0,
    )
    partial_offsets = row_block_idx * num_cols + col_offsets
    tl.store(
        partial_grad_weight_ptr + partial_offsets,
        partial,
        mask=col_offsets < num_cols,
    )


@triton.jit
def _kimi_rms_norm_gated_weight_grad_reduce_kernel(
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


@torch.library.triton_op("torchtitan::triton_kimi_rms_norm_gated", mutates_args={})
def _triton_kimi_rms_norm_gated_op(
    input: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    input = input.contiguous()
    gate = gate.contiguous()
    weight = weight.contiguous()
    num_cols = input.shape[-1]
    block_size = triton.next_power_of_2(num_cols)
    if block_size > _MAX_BLOCK_SIZE:
        raise ValueError(
            f"Triton KimiRMSNormGated supports at most {_MAX_BLOCK_SIZE} columns, "
            f"got {num_cols}"
        )
    num_rows = input.numel() // num_cols
    output = torch.empty_like(input)
    inverse_rms = torch.empty(num_rows, dtype=torch.float32, device=input.device)
    torch.library.wrap_triton(_kimi_rms_norm_gated_forward_kernel)[(num_rows,)](
        input,
        gate,
        weight,
        output,
        inverse_rms,
        num_cols=num_cols,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return output, inverse_rms


@torch.library.triton_op(
    "torchtitan::triton_kimi_rms_norm_gated_backward", mutates_args={}
)
def _triton_kimi_rms_norm_gated_backward_op(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    inverse_rms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output = grad_output.contiguous()
    num_cols = input.shape[-1]
    num_rows = input.numel() // num_cols
    block_size = triton.next_power_of_2(num_cols)

    grad_input = torch.empty_like(input)
    grad_gate = torch.empty_like(gate)
    torch.library.wrap_triton(_kimi_rms_norm_gated_input_gate_grad_kernel)[(num_rows,)](
        grad_output,
        input,
        gate,
        weight,
        inverse_rms,
        grad_input,
        grad_gate,
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
    torch.library.wrap_triton(_kimi_rms_norm_gated_weight_grad_partial_kernel)[
        (num_col_blocks, num_partials)
    ](
        grad_output,
        input,
        gate,
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
    torch.library.wrap_triton(_kimi_rms_norm_gated_weight_grad_reduce_kernel)[
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
    return grad_input, grad_gate, grad_weight


def _triton_kimi_rms_norm_gated_setup_context(ctx, inputs, output) -> None:
    input, gate, weight, _eps = inputs
    _output, inverse_rms = output
    ctx.save_for_backward(input, gate, weight, inverse_rms)


def _triton_kimi_rms_norm_gated_autograd_backward(
    ctx,
    grad_output: torch.Tensor,
    _grad_inverse_rms: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    input, gate, weight, inverse_rms = ctx.saved_tensors
    grad_input, grad_gate, grad_weight = _triton_kimi_rms_norm_gated_backward_op(
        grad_output.contiguous(),
        input,
        gate,
        weight,
        inverse_rms,
    )
    return grad_input, grad_gate, grad_weight, None


_triton_kimi_rms_norm_gated_op.register_autograd(
    _triton_kimi_rms_norm_gated_autograd_backward,
    setup_context=_triton_kimi_rms_norm_gated_setup_context,
)


def triton_kimi_rms_norm_gated(
    input: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply per-row RMS normalization, weight scaling, and sigmoid gating."""
    output, _inverse_rms = _triton_kimi_rms_norm_gated_op(
        input.contiguous(),
        gate.contiguous(),
        weight.contiguous(),
        eps,
    )
    return output


class TritonKimiRMSNormGated(KimiRMSNormGated):
    """Kimi K3 RMSNorm and sigmoid gate implemented by a fused Triton kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(KimiRMSNormGated.Config):
        pass

    def forward(
        self,
        x_THV: torch.Tensor,
        gate_THV: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not x_THV.is_cuda
            or x_THV.dtype not in _SUPPORTED_DTYPES
            or gate_THV.dtype not in _SUPPORTED_DTYPES
        ):
            return super().forward(x_THV, gate_THV)

        return triton_kimi_rms_norm_gated(
            x_THV,
            gate_THV,
            self.weight,
            self.eps,
        )


@override(
    target=KimiRMSNormGated.Config,
    exact=True,
    description="Fuse Kimi K3 RMSNorm, weight scaling, and sigmoid output gate.",
)
def triton_kimi_rmsnorm_gated(
    cfg: KimiRMSNormGated.Config,
) -> TritonKimiRMSNormGated.Config:
    sharding_config = cfg.sharding_config
    if sharding_config is not None:
        if sharding_config.state_shardings.get("weight") is None:
            raise ValueError(
                "Triton KimiRMSNormGated requires a weight sharding contract"
            )
        activation = attention_activation_placement()
        input_shardings = {
            "x_THV": activation,
            "gate_THV": activation,
        }
        sharding_config = replace(
            sharding_config,
            in_src_shardings=input_shardings,
            in_dst_shardings=input_shardings,
            out_src_shardings=activation,
            out_dst_shardings=activation,
            local_spmd=True,
        )
    return derive(
        cfg,
        TritonKimiRMSNormGated.Config,
        sharding_config=sharding_config,
    )
