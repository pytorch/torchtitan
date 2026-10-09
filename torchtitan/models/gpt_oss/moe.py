# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import torch
import torch_remat as remat
from torch import nn

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.linear import GroupedLinear


class GptOssGroupedLinear(GroupedLinear):
    """Grouped linear with GPT-OSS per-expert bias."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.bias = nn.Parameter(torch.empty(self.weight.shape[:-1]))

    def forward(self, input_RI: torch.Tensor, offsets_E: torch.Tensor) -> torch.Tensor:
        output_RO = super().forward(input_RI, offsets_E)
        # The bias add below reads the grouped_mm output with bare ops.
        remat.recompute_needs_tensor(output_RO)
        return add_grouped_bias(
            output_RO.flatten(1), self.bias.flatten(1), offsets_E
        ).view_as(output_RO)


# recompile_limit=16: w13 and w2 widths x (static, dynamic rows) x grad mode is 6 graphs
# before EP zero-row or eval shapes (default limit 8).
@local_compile("grouped_expert_bias", batch_invariant=True, recompile_limit=16)
def add_grouped_bias(
    output_RO: torch.Tensor, bias_EO: torch.Tensor, offsets_E: torch.Tensor
) -> torch.Tensor:
    """Add each routed row's expert bias: ``output_RO[r] + bias_EO[expert of row r]``.

    Rows past ``offsets_E[-1]`` are padding and get no bias.

    Example:
        offsets_E = [2, 3] (expert 0 owns rows 0-1, expert 1 row 2), 4 rows:
        out[0:2] += bias[0], out[2] += bias[1], out[3] unchanged.
    """
    if torch.compiler.is_compiling():
        # w13 and w2 call this with different widths; keep each one static.
        torch._dynamo.mark_static(output_RO, -1)
    row_ids_R = torch.arange(
        output_RO.shape[0], device=offsets_E.device, dtype=offsets_E.dtype
    )
    expert_ids_R = torch.searchsorted(offsets_E, row_ids_R, right=True)
    return _GroupedBiasAdd.apply(output_RO, bias_EO, expert_ids_R)


class _GroupedBiasAdd(torch.autograd.Function):
    """Gather-add of per-expert biases whose backward sums each expert's rows with a GEMM.

    The autograd backward of a gathered or repeat-interleaved bias is a scatter-add
    over all rows (sort-based or atomic, bf16 accumulation). A one-hot ``[E + 1, R]``
    matrix times the gradient sums the rows deterministically, accumulating in fp32.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        output_RO: torch.Tensor,
        bias_EO: torch.Tensor,
        expert_ids_R: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(expert_ids_R)
        ctx.num_experts = bias_EO.shape[0]
        ctx.bias_dtype = bias_EO.dtype
        # Padding rows (expert id E) index the appended zero row.
        padded_bias_EO = torch.cat((bias_EO, bias_EO.new_zeros(1, bias_EO.shape[-1])))
        return output_RO + padded_bias_EO[expert_ids_R].to(output_RO.dtype)

    @staticmethod
    def backward(ctx, grad_RO: torch.Tensor):  # pyrefly: ignore[bad-override]
        (expert_ids_R,) = ctx.saved_tensors
        one_hot_ER = grad_RO.new_zeros(ctx.num_experts + 1, grad_RO.shape[0])
        one_hot_ER.scatter_(0, expert_ids_R.long().unsqueeze(0), 1)
        grad_bias_EO = torch.mm(one_hot_ER, grad_RO)[: ctx.num_experts]
        return grad_RO, grad_bias_EO.to(ctx.bias_dtype), None
