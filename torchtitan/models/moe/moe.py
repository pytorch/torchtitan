# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.deepep.fused_activation import fused_silu_gate_prob

from torchtitan.distributed.deepep.utils import DeepEPTokenDispatcher

from .utils import indices_padding_wrapper


def moe_init_std(dim_in: int, n_layers: int) -> float:
    return (2 / (dim_in * n_layers)) ** 0.5


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1
    shared_gate: bool = False

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    load_balance_coeff: float | None = 1e-3

    _debug_force_load_balance: bool = False

    # use deepep and fused all-to-all communication
    use_deepep: bool = False
    # if True, we force each experts get same amount of token via round-robin


# can be used as dense FFN layer or shared experts in MoE layers
class FeedForward(nn.Module):
    """
    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


# NOTE: keeping this for-loop implementation for comparison
#       and readability, may remove later
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert = num_tokens_per_expert.tolist()

    # side-effect code due to the usage of generate_permute_indices
    num_padding = x.shape[0] - sum(num_tokens_per_expert)

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    x = torch.split(
        x[: sum(num_tokens_per_expert)],
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor | None,
    routed_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Run grouped matrix multiplication for MoE experts with SwiGLU activation.

    Args:
        w1: First expert weights [num_experts, hidden_dim, dim]
        w2: Second expert weights [num_experts, dim, hidden_dim]
        w3: Gate expert weights [num_experts, hidden_dim, dim]
        x: Input tensor [total_tokens, dim]
        num_tokens_per_expert: Number of tokens per expert [num_experts]
        routed_prob: Routing probabilities [total_tokens, 1] or [total_tokens] (optional)
                     When provided with fused kernel, enables fused silu-gate-prob computation.

    Returns:
        Output tensor [total_tokens, dim]
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # ============================================================================
    # FUSED ACTIVATION KERNEL (Triton-optimized)
    # ============================================================================
    # When routed_prob is provided, we use the fused kernel that combines:
    #   - SiLU activation: silu(x @ w1)
    #   - Gate multiplication: * (x @ w3)
    #   - Routing probability scaling: * prob
    #
    # This provides ~3.2x speedup over the non-fused version by:
    #   1. Reducing memory bandwidth (single kernel launch vs 3 separate ops)
    #   2. Avoiding intermediate tensor allocations
    #   3. Fusing dtype conversions
    #
    # Performance (Qwen3-30B-A3B, 16384 tokens, hidden=768):
    #   - Non-fused: 0.0967 ms
    #   - Fused:     0.0308 ms (3.2x faster)
    #   - Per-step savings: ~2.79 ms (48 layers)
    #
    # See: torchtitan/distributed/deepep/fused_activation.py for kernel details
    # ============================================================================
    if routed_prob is not None:
        # Compute x @ w1 and x @ w3 (no activation yet)
        x1 = torch._grouped_mm(
            x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
        )
        x3 = torch._grouped_mm(
            x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
        )

        # Fused: silu(x1) * x3 * prob (single Triton kernel)
        h = fused_silu_gate_prob(x1, x3, routed_prob)

        # Final projection: h @ w2
        out = torch._grouped_mm(
            h, w2.bfloat16().transpose(-2, -1), offs=offsets
        ).type_as(x)
    else:
        # ============================================================================
        # NON-FUSED PATH (original implementation)
        # ============================================================================
        # Used when routed_prob is not provided (standard EP path without DeepEP).
        # The routing score multiplication happens separately in MoE.forward().
        # ============================================================================
        h = F.silu(
            torch._grouped_mm(
                x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
            )
        )
        h = h * torch._grouped_mm(
            x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
        )
        out = torch._grouped_mm(
            h, w2.bfloat16().transpose(-2, -1), offs=offsets
        ).type_as(x)

    return out


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
        deepep_dispatcher: DeepEPTokenDispatcher = None,
        score_before_experts: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = use_grouped_mm
        self.deepep_dispatcher = deepep_dispatcher
        # NOTE(phuc): fix compatibility with ExpertParallelDeepEP
        # When DeepEP is used, it passes routed_prob to forward() which needs to be multiplied
        # with the input/output. This flag controls whether to apply the scaling before or after
        # the expert computation. See torchtitan-amd implementation for reference.
        self.score_before_experts = score_before_experts

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        routed_prob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        # ============================================================================
        # NOTE(phuc): CRITICAL: Routing Score Multiplication for DeepEP
        # ============================================================================
        # This code multiplies expert outputs by their routing probabilities (top_scores).
        # DO NOT COMMENT OUT - removing this causes an 8x gradient explosion bug.
        #
        # WHY THIS IS NECESSARY:
        # In MoE with top_k routing, each token is sent to K experts. The final output
        # should be a weighted sum: output = sum(expert_i(x) * prob_i) for i in top_k
        #
        # Standard EP vs DeepEP paths handle this differently:
        #
        # STANDARD EP PATH (MoE.forward):
        #   1. Router computes top_scores [bs*slen, top_k]
        #   2. TokenReorderer sorts tokens by expert
        #   3. Experts compute outputs
        #   4. MoE.forward() multiplies: routed_output *= top_scores  <-- SCALING HERE
        #   5. scatter_add combines outputs back to original positions
        #
        # DEEPEP PATH (MoE.forward:
        #   1. Router computes top_scores [bs*slen, top_k]
        #   2. dispatch_preprocess stores scores in dispatcher
        #   3. ExpertParallelDeepEP._token_dispatch returns routed_prob
        #   4. GroupedExperts.forward() must multiply: out *= routed_prob  <-- HERE
        #   5. DeepEP combine operation aggregates results
        #
        # WITHOUT THIS MULTIPLICATION:
        #   - Each token's output from K experts is summed without weighting
        #   - Output magnitude is ~K times larger (K=8 for Qwen3-30B-A3B)
        #   - Gradients are ~8x larger, causing training instability
        #   - Gate gradients become zero (router doesn't learn)
        #
        # VERIFIED: test_full_comparison_table.py shows:
        #   - With scaling:    grad_norm_ratio = 1.00x ✓
        #   - Without scaling: grad_norm_ratio = 7.66x ✗ (≈ top_k)
        #
        # Performance note: This adds ~20% overhead but is mathematically required.
        # ============================================================================
        if (
            self.deepep_dispatcher is not None
            and routed_prob is not None
            and self.score_before_experts
        ):
            x = (x.to(torch.float32) * routed_prob.reshape(-1, 1)).to(x.dtype)

        if self.use_grouped_mm:
            # NOTE: If EP is not used, we need to pad the indices
            #       to prepare for grouped_mm;
            #       otherwise, EP will handle the padding.
            if (
                not isinstance(self.w1, DTensor)
                or "ep" not in self.w1.device_mesh.mesh_dim_names
            ):
                run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)
            else:
                run_experts_fn = _run_experts_grouped_mm

            # ============================================================================
            # FUSED KERNEL PATH (DeepEP + score_before_experts=False)
            # ============================================================================
            # When using DeepEP with score_before_experts=False (default), we pass
            # routed_prob to _run_experts_grouped_mm to enable the fused Triton kernel.
            #
            # The fused kernel combines: silu(x @ w1) * (x @ w3) * prob
            # into a single kernel, providing ~3.2x speedup over separate operations.
            #
            # This replaces the separate multiplication that was done after experts:
            #   OLD: out = (out.float() * routed_prob.reshape(-1, 1)).to(out.dtype)
            #   NEW: Fused inside _run_experts_grouped_mm via fused_silu_gate_prob()
            #
            # For other paths (standard EP, score_before_experts=True), routed_prob
            # is not passed, and the non-fused path is used.
            # ============================================================================
            use_fused_kernel = (
                self.deepep_dispatcher is not None
                and routed_prob is not None
                and not self.score_before_experts
            )

            if use_fused_kernel:
                out = run_experts_fn(w1, w2, w3, x, num_tokens_per_expert, routed_prob)
            else:
                out = run_experts_fn(w1, w2, w3, x, num_tokens_per_expert)
        else:
            out = _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

        # ============================================================================
        # OLD CODE (commented out - replaced by fused kernel above)
        # ============================================================================
        # NOTE(phuc): Apply routing scores AFTER expert computation if score_before_experts=False
        # Only for DeepEP: routed_prob is passed from ExpertParallelDeepEP._token_dispatch
        #
        # This separate multiplication is now FUSED into _run_experts_grouped_mm when
        # use_fused_kernel=True. The fused kernel combines silu + gate + prob multiply
        # into a single Triton kernel for ~3.2x speedup.
        #
        # KEEP THIS COMMENTED CODE for reference and rollback if needed:
        # ============================================================================
        # if (
        #     self.deepep_dispatcher is not None
        #     and routed_prob is not None
        #     and not self.score_before_experts
        # ):
        #     out = (out.to(torch.float32) * routed_prob.reshape(-1, 1)).to(out.dtype)

        return out

    def init_weights(self, init_std: float, n_layers: int):
        std_in = moe_init_std(self.w1.shape[-1], n_layers)
        std_out = moe_init_std(self.w2.shape[0], n_layers)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=std_in)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=std_in)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=std_out)


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid"]): Whether to use sigmoid or softmax for router scores.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
        _debug_force_load_balance: bool = False,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self._debug_force_load_balance = _debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.size(0)
        # Round-robin indices with exact balance
        selected_experts_indices = (
            torch.arange(
                n_tokens * self.top_k, device=scores.device, dtype=torch.int64
            ).reshape(n_tokens, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        num_tokens_per_expert = torch.clamp(num_tokens_per_expert, min=8)

        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float, n_layers: int):
        temp_weight = torch.empty_like(self.gate.weight)
        nn.init.normal_(temp_weight, mean=0.0, std=1.0)

        # TODO(phuc): change to torch.linalg.norm
        # due to TOR101 Use of deprecated function torch.norm
        row_norms = torch.norm(temp_weight, dim=1, keepdim=True)
        temp_weight = temp_weight / row_norms.clamp(min=1e-6)  # avoid divide by 0

        std = moe_init_std(self.gate.weight.shape[1], n_layers)
        self.gate.weight.data = temp_weight * std


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size * seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()

        num_experts = moe_args.num_experts

        self.use_deepep = moe_args.use_deepep
        if self.use_deepep:
            self.deepep_dispatcher = DeepEPTokenDispatcher(
                moe_router_topk=moe_args.top_k,
                num_moe_experts=num_experts,
            )

        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
            deepep_dispatcher=self.deepep_dispatcher
            if self.use_deepep is True
            else None,
            # NOTE(phuc): fix ExpertParallelDeepEP compatibility
            # This ensures that GroupedExperts knows whether to apply routing scores before or after
            # expert computation when routed_prob is passed to forward()
            score_before_experts=moe_args.score_before_experts,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
            _debug_force_load_balance=moe_args._debug_force_load_balance,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=moe_args.top_k)
        self.shared_experts = (
            FeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts)
            if moe_args.num_shared_experts > 0
            else None
        )
        self.shared_gate = (
            nn.Linear(dim, 1, bias=False) if moe_args.shared_gate else None
        )
        self.score_before_experts = moe_args.score_before_experts

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None

        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # top_scores and selected_experts_indices shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        if self.use_deepep:
            top_scores = top_scores.float()
            self.experts.deepep_dispatcher.dispatch_preprocess(
                top_scores, selected_experts_indices
            )
            # shape (bs*slen*top_k, dim)
            routed_output = self.experts(x, num_tokens_per_expert)
            # shared expert
            if self.shared_experts is not None:
                out = self.shared_experts(x)
            else:
                out = torch.zeros_like(x)
            out = routed_output + out
            out = out.reshape(bs, slen, dim)
            return out
        else:
            # top_scores and token_indices_experts_sorted shape (bs*slen*top_k,)
            # num_tokens_per_expert shape (num_experts,)
            # NOTE: the reason we need to compute num_tokens_per_expert again is:
            #       1st computation in router is to update self.tokens_per_expert
            #       which would be the same across all TP ranks.
            #       2nd computation in reorderer is for the actual routing and experts computation
            #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
            #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
            (
                top_scores_experts_sorted,
                token_indices_experts_sorted,
                num_tokens_per_expert,
            ) = self.reorderer(top_scores, selected_experts_indices)

            # shape (bs*slen*top_k, dim)
            token_indices_experts_sorted = token_indices_experts_sorted.reshape(
                -1, 1
            ).expand(-1, dim)

            # shape (bs*slen*top_k, dim)
            routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)

            if self.score_before_experts:
                routed_input = (
                    routed_input.to(torch.float32)
                    * top_scores_experts_sorted.reshape(-1, 1)
                ).to(x.dtype)

            # shape (bs*slen*top_k, dim)
            routed_output = self.experts(routed_input, num_tokens_per_expert)

            # shared expert
            # Note: we execute the shared expert before scoring the output of the routed expert
            # to "implicitly" overlap the shared expert compute with token combine communication
            if self.shared_experts is not None:
                out = self.shared_experts(x)
            else:
                out = torch.zeros_like(x)

            if not self.score_before_experts:
                routed_output = (
                    routed_output.to(torch.float32)
                    * top_scores_experts_sorted.reshape(-1, 1)
                ).to(x.dtype)

            out = out.scatter_add(
                dim=0, index=token_indices_experts_sorted, src=routed_output
            )
            out = out.reshape(bs, slen, dim)
            return out

    def init_weights(self, init_std: float, buffer_device: torch.device, n_layers: int):
        self.experts.init_weights(init_std, n_layers)
        self.router.init_weights(init_std, n_layers)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)
            if self.shared_gate is not None:
                nn.init.trunc_normal_(
                    self.shared_gate.weight,
                    mean=0.0,
                    std=moe_init_std(self.shared_gate.weight.shape[1], n_layers),
                )

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
