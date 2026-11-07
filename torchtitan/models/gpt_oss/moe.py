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
from torch.utils.checkpoint import checkpoint

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.activation import BinaryActivationFn
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts

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
        with_bias: bool = True,
    ) -> torch.Tensor:
        output_RO = super().forward(input_RI, offsets_E, scores_R=scores_R)
        if not with_bias:
            # GptOssRoutedExperts adds w13's bias inside its activation region.
            return output_RO
        flat_output_RO = output_RO.flatten(1)
        # Mark rows dynamic from the first call (scores_R has the same rows). Without it,
        # PP fails in backward:
        # 1. Microbatch A's forward compiles a graph for A's row count.
        # 2. Microbatch B's forward has other rows and compiles a dynamic-rows graph.
        # 3. A's backward replays the region on B's graph, which saves other tensors.
        # Dynamo can't trace this call, so skip it when the caller itself is compiled.
        if not torch.compiler.is_compiling():
            torch._dynamo.maybe_mark_dynamic(flat_output_RO, 0)
            if scores_R is not None:
                torch._dynamo.maybe_mark_dynamic(scores_R, 0)
        # Match the matmul's policy: replaying even this cheap add after a saved
        # matmul would force retention of its full routed output.
        add_bias = remat.region(
            add_grouped_bias,
            self.remat_region_name("bias_add"),
            recompute=self.remat_should_recompute("grouped_mm"),
        )
        # Pass a copy of the bias. Under PP, FSDP2 keeps an fp32 .grad on the bf16 bias between
        # microbatches, and Dynamo raises when it compiles a parameter input whose .grad has
        # another dtype. The copy costs one [E, O] kernel per projection per forward.
        # TODO: pass self.bias once Dynamo copies a tensor's grad_dtype before its .grad
        # (issue draft: felipemello1/pytorch#23).
        biased_RO = add_bias(
            flat_output_RO, self.bias.flatten(1).clone(), offsets_E, scores_R=scores_R
        )
        return biased_RO.view_as(output_RO)


# recompile_limit=16 covers the graphs per process: 3 per width (w13, w2), one for 2+ rows
# and one each for 0 and 1 rows (EP ranks with few tokens), twice if the process runs both
# grad and no_grad. Up to 12.
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
        # Eager autograd saves the gathered [R, O] bias for the score grad. The compiled
        # region (the default) saves only the [E, O] bias and gathers again.
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
        # Padding rows (expert id E) get zero. torch.where keeps this gather inside its
        # consumer's kernel. A padded copy of the bias (cat with a zero row) adds 2 kernels
        # per call in the w13_bias_activation region.
        in_range_R1 = (expert_ids_R < bias_EO.shape[0]).unsqueeze(-1)
        gathered_RO = bias_EO[expert_ids_R.clamp(max=bias_EO.shape[0] - 1)]
        return torch.where(in_range_R1, gathered_RO, 0)

    @staticmethod
    def backward(ctx, grad_RO: torch.Tensor):  # pyrefly: ignore[bad-override]
        (expert_ids_R,) = ctx.saved_tensors
        one_hot_ER = grad_RO.new_zeros(ctx.num_experts + 1, grad_RO.shape[0])
        one_hot_ER.scatter_(0, expert_ids_R.long().unsqueeze(0), 1)
        return torch.mm(one_hot_ER, grad_RO)[: ctx.num_experts], None


class GptOssRoutedExperts(RoutedExperts):
    """Routed experts that add w13's bias inside the activation region."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def _w13_and_activation(
        self, routed_input_RD: torch.Tensor, offsets_E: torch.Tensor
    ) -> torch.Tensor:
        gate_up_R2F = self.w13(routed_input_RD, offsets_E, with_bias=False)
        remat.recompute_needs_tensor(gate_up_R2F)
        # Marking rows dynamic from the first call saves one recompile. The bias copy is for
        # PP + FSDP2: see GptOssGroupedLinear.forward, whose TODO also covers this copy.
        if not torch.compiler.is_compiling():
            torch._dynamo.maybe_mark_dynamic(gate_up_R2F, 0)
        return add_bias_and_activate(
            gate_up_R2F, self.w13.bias.clone(), offsets_E, self.activation_fn
        )


# Graphs per process: one for 2+ rows and one each for 0 and 1 rows (EP ranks with few
# tokens), twice if the process runs both grad and no_grad. Up to 6. recompile_limit=16
# leaves room for more: under fullgraph, the first graph past the limit raises.
# The options match fused_binary_activation, whose __call__ this region inlines.
@local_compile(
    "w13_bias_activation",
    batch_invariant=True,
    recompile_limit=16,
    options={"max_pointwise_cat_inputs": 0, "max_complex_pointwise_cat_inputs": 0},
)
def add_bias_and_activate(
    gate_up_R2F: torch.Tensor,
    bias_E2F: torch.Tensor,
    offsets_E: torch.Tensor,
    activation_fn: BinaryActivationFn,
) -> torch.Tensor:
    """Return ``activation_fn(gate_up_R2F + bias_E2F[expert of row])``, both in one region.

    Example:
        offsets_E = [2, 3] (expert 0 owns rows 0-1, expert 1 row 2), 4 rows:
        rows 0-1 add bias_E2F[0], row 2 adds bias_E2F[1], row 3 (padding) adds nothing,
        then activation_fn maps each row's [2, F] gate and up to [F].
    """

    def biased_activation(
        gate_up_R2F: torch.Tensor, bias_E2F: torch.Tensor
    ) -> torch.Tensor:
        gate_up_RO = gate_up_R2F.flatten(1)
        # Outside a Dynamo trace, add_grouped_bias compiles on its own. Mark its rows dynamic
        # for PP, as GptOssGroupedLinear.forward does.
        if not torch.compiler.is_compiling():
            torch._dynamo.maybe_mark_dynamic(gate_up_RO, 0)
        biased_RO = add_grouped_bias(gate_up_RO, bias_E2F.flatten(1), offsets_E)
        return activation_fn(biased_RO.view_as(gate_up_R2F), offsets=offsets_E)

    # The checkpoint steers the compiler's partitioner, so it helps only when Dynamo traces
    # this code (this region or an outer compile). Otherwise run the two regions as they are:
    # a checkpoint would rerun both in backward and fuse nothing.
    if not torch.compiler.is_compiling():
        return biased_activation(gate_up_R2F, bias_E2F)
    # Backward recomputes this function, so the region saves only gate_up_R2F and never writes
    # the biased [R, 2F]. The add then fuses into one forward and one backward kernel.
    # The activation reads the add's unrounded fp32 sum. The region-off path rounds it to bf16.
    # TODO: add the bias inside w13's GEMM once torch._grouped_mm accepts one (today it raises
    # "Bias not supported yet"; issue draft: felipemello1/pytorch#25). The recompute and this
    # checkpoint then go away.
    return checkpoint(biased_activation, gate_up_R2F, bias_E2F, use_reentrant=False)
