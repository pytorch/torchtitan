# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat
from torch import nn

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.linear import GroupedLinear

# Shape suffix legend: R = routed rows, E = experts, I = input features,
# O = flattened output features, including stacked projections.


class GptOssGroupedLinear(GroupedLinear):
    """Grouped linear with GPT-OSS per-expert bias."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.bias = nn.Parameter(torch.empty(self.weight.shape[:-1]))

    def forward(
        self,
        input_RI: torch.Tensor,
        offsets_E: torch.Tensor,
        *,
        scores_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_RO = super().forward(input_RI, offsets_E, scores_R=scores_R)
        # Match the matmul's policy: replaying even this cheap add after a saved
        # matmul would force retention of its full routed output.
        add_bias = remat.region(
            add_grouped_bias,
            self.remat_region_name("bias_add"),
            recompute=self.remat_should_recompute("grouped_mm"),
        )
        biased_RO = add_bias(
            output_RO.flatten(1), self.bias.flatten(1), offsets_E, scores_R=scores_R
        )
        return biased_RO.view_as(output_RO)


# TODO: add w13's bias inside the fused_binary_activation region (RoutedExperts must then
# pass the bias to the activation). Without grad: one kernel instead of two. With grad it
# only pays if SwiGLU's backward recomputes the add, so SwiGLU would read the fp32 sum
# instead of today's bf16-rounded w13 output: not bitwise.
#
# recompile_limit=16: w13 and w2 widths x (static, dynamic rows) is 4 graphs in grad mode;
# no_grad adds 2 (it starts dynamic); 0-row and 1-row calls (EP ranks with few tokens) each
# add one graph per width and mode (+8): 14 at most (default limit 8).
@local_compile("grouped_expert_bias", batch_invariant=True, recompile_limit=16)
def add_grouped_bias(
    output_RO: torch.Tensor,
    bias_EO: torch.Tensor,
    offsets_E: torch.Tensor,
    *,
    scores_R: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add each routed row's expert bias: ``output_RO[r] + bias_EO[expert of row r]``.

    Rows past ``offsets_E[-1]`` are padding and get no bias. With ``scores_R``
    (router scores absorbed into W2's input), each row's bias is scaled by its score.

    Example:
        offsets_E = [2, 3] (expert 0 owns rows 0-1, expert 1 row 2), 4 rows:
        out[0:2] += bias[0], out[2] += bias[1], out[3] unchanged.
    """
    # w13 and w2 call this with different widths; keep each one static.
    torch._dynamo.mark_static(output_RO, -1)
    row_ids_R = torch.arange(
        output_RO.shape[0], device=offsets_E.device, dtype=offsets_E.dtype
    )
    expert_ids_R = torch.searchsorted(offsets_E, row_ids_R, right=True)
    bias_RO = _GatherGroupedBias.apply(bias_EO, expert_ids_R)
    if scores_R is not None:
        bias_RO = bias_RO * scores_R.reshape(-1, 1).to(torch.float32)
    return output_RO + bias_RO.to(output_RO.dtype)


@spmd.register_local_autograd_function
class _GatherGroupedBias(torch.autograd.Function):
    """Gather of per-expert biases whose backward sums each expert's rows with a GEMM.

    The autograd backward of a gathered or repeat-interleaved bias is a scatter-add
    over all rows (sort-based or atomic, bf16 accumulation). A one-hot ``[E + 1, R]``
    matrix times the gradient sums the rows deterministically, accumulating in fp32.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, bias_EO: torch.Tensor, expert_ids_R: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(expert_ids_R)
        ctx.num_experts = bias_EO.shape[0]
        # Padding rows (expert id E) index the appended zero row.
        padded_bias_EO = torch.cat((bias_EO, bias_EO.new_zeros(1, bias_EO.shape[-1])))
        return padded_bias_EO[expert_ids_R]

    @staticmethod
    def backward(ctx, grad_RO: torch.Tensor):  # pyrefly: ignore[bad-override]
        (expert_ids_R,) = ctx.saved_tensors
        one_hot_ER = grad_RO.new_zeros(ctx.num_experts + 1, grad_RO.shape[0])
        one_hot_ER.scatter_(0, expert_ids_R.long().unsqueeze(0), 1)
        return torch.mm(one_hot_ER, grad_RO)[: ctx.num_experts], None
