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
from torch.distributed.tensor import DTensor, Partial

from torchtitan.config import Configurable
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear

from torchtitan.protocols.module import Module

from .token_dispatcher import LocalTokenDispatcher


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
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    # NOTE: x is not sliced because padding was removed in #2774, so
    # sum(num_tokens_per_expert) == x.shape[0] always holds.
    x_splits = torch.split(
        x,
        split_size_or_sections=num_tokens_per_expert_list,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = F.silu(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        use_grouped_mm: bool = True
        token_dispatcher: LocalTokenDispatcher.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.use_grouped_mm = config.use_grouped_mm
        self.token_dispatcher = config.token_dispatcher.build()

    def _experts_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine."""
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = self.w2.to_local()
            # pyrefly: ignore [missing-attribute]
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.use_grouped_mm:
            return _run_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)
        else:
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

    def forward(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        shared_experts is passed to combine() where it overlaps with the async
        combine all-to-all (NCCL stream) or async DeepEP combine.
        """
        routed_input, num_tokens_local, metadata = self.token_dispatcher.dispatch(
            x, top_scores, selected_experts_indices
        )
        routed_output = self._experts_forward(routed_input, num_tokens_local)
        return self.token_dispatcher.combine(routed_output, metadata, x, shared_experts)


def _sequence_wise_aux_loss(
    scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    bs: int,
    slen: int,
    top_k: int,
    aux_loss_weight: float,
) -> torch.Tensor:
    """Sequence-wise auxiliary load-balance loss (DeepSeek-V3 Eqs 17-20)."""
    num_experts = scores.size(-1)
    scores_per_seq = scores.view(bs, slen, num_experts)
    denom = scores_per_seq.sum(dim=-1, keepdim=True) + 1e-20
    probs_per_seq = scores_per_seq / denom
    p_i = probs_per_seq.mean(dim=1)
    indices_per_seq = selected_experts_indices.view(bs, -1)
    offset = torch.arange(bs, device=indices_per_seq.device).unsqueeze(1) * num_experts
    flat_indices = (indices_per_seq + offset).reshape(-1)
    counts = torch.bincount(flat_indices.long(), minlength=bs * num_experts)
    counts = counts.reshape(bs, num_experts).to(dtype=scores.dtype)
    f_i = counts * (num_experts / (top_k * slen))
    return (f_i * p_i).sum(dim=1).mean() * aux_loss_weight


def _batch_wise_aux_loss(
    scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    bs: int,
    slen: int,
    top_k: int,
    aux_loss_weight: float,
) -> torch.Tensor:
    """Batch-wise auxiliary load-balance loss."""
    num_experts = scores.size(-1)
    total_tokens = scores.size(0)
    num_tokens_per_expert = torch.histc(
        selected_experts_indices.view(-1).float(),
        bins=num_experts,
        min=0,
        max=num_experts,
    )
    p_i = scores.mean(dim=0)
    f_i = num_tokens_per_expert.to(scores.dtype) * (
        num_experts / (top_k * total_tokens)
    )
    return (f_i * p_i).sum() * aux_loss_weight


class _AuxLossBase(torch.autograd.Function):
    """Injects auxiliary load-balance loss gradients at the router scores level.

    Identity in forward (returns ``top_scores`` unchanged). In backward,
    computes ``d(aux_loss)/d(scores)`` via ``torch.func.grad`` and adds it
    to ``scores``'s gradient. ``top_scores`` is a pass-through so this node
    remains in the autograd graph.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx: torch.autograd.function.FunctionCtx,
        top_scores: torch.Tensor,
        scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        bs: int,
        slen: int,
        top_k: int,
        aux_loss_weight: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(scores, selected_experts_indices)
        ctx.bs = bs  # pyrefly: ignore [missing-attribute]
        ctx.slen = slen  # pyrefly: ignore [missing-attribute]
        ctx.top_k = top_k  # pyrefly: ignore [missing-attribute]
        ctx.aux_loss_weight = aux_loss_weight  # pyrefly: ignore [missing-attribute]
        return top_scores

    @staticmethod
    def _backward_impl(loss_fn, ctx, grad_top_scores):
        (
            scores,
            selected_experts_indices,
        ) = ctx.saved_tensors
        # torch.func.grad avoids the graph break that torch.autograd.grad causes under torch.compile
        aux_grad = torch.func.grad(loss_fn)(
            scores,
            selected_experts_indices,
            ctx.bs,
            ctx.slen,
            ctx.top_k,
            ctx.aux_loss_weight,
        )
        return grad_top_scores, aux_grad, None, None, None, None, None


class _SequenceWiseAuxLoss(_AuxLossBase):
    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_top_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:
        return _AuxLossBase._backward_impl(
            _sequence_wise_aux_loss, ctx, grad_top_scores
        )


class _BatchWiseAuxLoss(_AuxLossBase):
    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_top_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:
        return _AuxLossBase._backward_impl(_batch_wise_aux_loss, ctx, grad_top_scores)


class TokenChoiceTopKRouter(Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Optionally supports node-limited (group-limited) routing where experts are divided into groups
    (e.g., by node), and only num_limited_groups groups are considered before selecting top_k experts.
    This reduces cross-node communication in distributed settings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: Linear.Config
        num_expert_groups: int | None = None  # must be a divisor of num_experts
        num_limited_groups: int | None = None
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid"] = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance

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

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
            and set expert scores in non-selected groups as -inf

        Args:
            scores_for_choice: Router scores with expert_bias (if any), shape (bs*slen, num_experts)

        Returns:
            scores_for_choice: shape (bs*slen, num_experts)
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice.view(
            -1, self.num_expert_groups, experts_per_group
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).view(-1, self.num_experts)

        return scores_for_choice

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - scores (torch.Tensor):
                    Full router scores for all experts with shape ``(bs*slen, num_experts)``.
                    Returned for load-balancing aux loss computation.
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        # Compute gate in float32 to help stability of expert load balancing.
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        # scored is already float32 from the autocast above.
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        # Apply node-limited routing if configured
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
        _, selected_experts_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = scores.gather(dim=1, index=selected_experts_indices)

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

        return scores, top_scores, selected_experts_indices, num_tokens_per_expert


class MoELoadBalanceAuxLoss(Configurable):
    """MoE auxiliary load-balance loss.

    Injects aux loss gradients into router scores without changing the model
    output (PP-safe). Call instance with router outputs to apply.

    Subclass to select loss variant (sequence-wise or batch-wise).
    """

    _autograd_fn: type[_AuxLossBase]

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        weight: float = 0.0
        """Weight for the auxiliary load-balance loss. 0 disables it."""
        top_k: int = 1
        """Number of experts per token. Must match the router's top_k."""
        global_batch_size: int | None = None
        """Global batch size to normalize aux loss across all batches/microbatches.
        Torchtitan generally uses sum aggregation & global_bs division.
        Because BatchWiseAuxLoss doesn't commute over microbatches, we ban
        batch-wise + PP usage in any case."""

    def __init__(self, config: Config):
        if config.weight < 0:
            raise ValueError(f"aux_loss.weight must be >= 0, got {config.weight}")
        self.weight = config.weight
        self.global_batch_size = config.global_batch_size
        self.top_k = config.top_k

    def __call__(
        self,
        top_scores: torch.Tensor,
        scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        bs: int,
        slen: int,
    ) -> torch.Tensor:
        if self.weight == 0:
            return top_scores
        global_bs = self.global_batch_size or bs
        scaled_weight = self.weight * bs / global_bs
        return self._autograd_fn.apply(
            top_scores,
            scores,
            selected_experts_indices,
            bs,
            slen,
            self.top_k,
            scaled_weight,
        )


class SequenceWiseAuxLoss(MoELoadBalanceAuxLoss):
    """Sequence-wise auxiliary load-balance loss (DeepSeek-V3 Eqs 17-20)."""

    _autograd_fn = _SequenceWiseAuxLoss

    @dataclass(kw_only=True, slots=True)
    class Config(MoELoadBalanceAuxLoss.Config):
        pass


class BatchWiseAuxLoss(MoELoadBalanceAuxLoss):
    """Batch-wise auxiliary load-balance loss."""

    _autograd_fn = _BatchWiseAuxLoss

    @dataclass(kw_only=True, slots=True)
    class Config(MoELoadBalanceAuxLoss.Config):
        pass


class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments
    2. GroupedExperts.forward() handles:
       a. dispatch (TokenDispatcher) — reorder tokens by expert assignment.
          With EP, also performs all-to-all communication to send tokens
          to expert-owning ranks.
       b. expert computation
       c. combine (TokenDispatcher) — reverse the dispatch reordering.
          With EP, starts async communication (NCCL all-to-all or DeepEP
          combine), runs shared_experts in parallel, then forces sync
          (scatter_add for NCCL AllToAll, sync_combine for DeepEP) and
          produces final output.
          Without EP (LocalTokenDispatcher), no communication is needed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        experts: GroupedExperts.Config
        router: TokenChoiceTopKRouter.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None
        aux_loss: MoELoadBalanceAuxLoss.Config | None = None

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.experts = config.experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
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
        # Convert DTensor to local tensor for MoE-internal computation.
        # grad_placements=(Partial(),) ensures x.grad is Partial on the tp_mesh
        # in backward, so gradient reduction (reduce-scatter from Partial to
        # Shard(1)) happens once at the MoE boundary rather than being
        # duplicated inside the MoE.
        #
        # Why grad(x) is Partial on the tp_mesh across all parallelism:
        # - TP only / TP+EP with ETP=TP: TP-sharded expert weights (Colwise on
        #   w1/w3, Rowwise on w2) produce Partial output gradients.
        # - TP+EP with ETP=1: each TP rank processes a disjoint token subset
        #   (via sequence-parallel token splitting in AllToAllTokenDispatcher),
        #   so grad(x) is non-zero only at each rank's token positions (Partial).
        #
        # This holds for all MoE components (router.gate, routed experts, shared
        # experts) and regardless of score_before_experts.
        if isinstance(x, DTensor):
            assert (
                x.device_mesh.ndim == 1
            ), f"Expected 1D mesh, got {x.device_mesh.ndim}D mesh"
            assert x.device_mesh.mesh_dim_names == (
                "tp",
            ), f"Expected TP mesh, got mesh_dim_names={x.device_mesh.mesh_dim_names}"
            x = x.to_local(grad_placements=(Partial(),))
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # scores shape (bs*slen, num_experts)
        # top_scores and selected_experts_indices shape (bs*slen, top_k)
        # num_tokens_per_expert shape (num_experts,)
        (
            scores,
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        if self.training and self.aux_loss is not None:
            top_scores = self.aux_loss(
                top_scores, scores, selected_experts_indices, bs, slen
            )

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        out = self.experts(
            x,
            top_scores,
            selected_experts_indices,
            shared_experts=self.shared_experts,
        )

        return out.reshape(bs, slen, dim)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert isinstance(buffer_device, torch.device)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
