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

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

from .token_dispatcher import DeepEPTokenDispatcher, LocalTokenDispatcher

# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, S = sequence length, D = model dimension,
#   H = hidden (FFN intermediate) dimension, E = num experts,
#   K = top-k, N = num tokens (B*S flattened), T = routed tokens


class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
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
        self.token_dispatcher = config.token_dispatcher.build()

    def _experts_forward(
        self,
        x_TD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine."""
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1_EHD = self.w1.to_local()
            # pyrefly: ignore [missing-attribute]
            w2_EDH = self.w2.to_local()
            # pyrefly: ignore [missing-attribute]
            w3_EHD = self.w3.to_local()
        else:
            w1_EHD = self.w1
            w2_EDH = self.w2
            w3_EHD = self.w3

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)

        h_TH = F.silu(
            torch._grouped_mm(
                x_TD.bfloat16(),
                w1_EHD.bfloat16().transpose(-2, -1),
                offs=offsets_E,
            )
        )
        h_TH = h_TH * torch._grouped_mm(
            x_TD.bfloat16(),
            w3_EHD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        return torch._grouped_mm(
            h_TH, w2_EDH.bfloat16().transpose(-2, -1), offs=offsets_E
        ).type_as(x_TD)

    def forward(
        self,
        x_BSD: torch.Tensor,
        top_scores_BSK: torch.Tensor,
        selected_experts_indices_BSK: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        The flatten to 2-D happens here; the caller (``MoE.forward``)
        reshapes the routed-expert output back to 3-D.

        When parallelized, ``local_map`` (from ``sharding_config``) handles
        DTensor→local conversion on entry and local→DTensor(Partial) wrapping
        on exit. The forward body operates on plain local tensors.
        """
        x_ND = x_BSD.reshape(-1, x_BSD.size(-1))
        top_scores_NK = top_scores_BSK.reshape(-1, top_scores_BSK.size(-1))
        selected_experts_indices_NK = selected_experts_indices_BSK.reshape(
            -1, selected_experts_indices_BSK.size(-1)
        )
        routed_input_TD, num_tokens_local_E, metadata = (
            self.token_dispatcher.dispatch(
                x_ND, top_scores_NK, selected_experts_indices_NK
            )
        )
        routed_output_TD = self._experts_forward(
            routed_input_TD, num_tokens_local_E
        )
        return self.token_dispatcher.combine(routed_output_TD, metadata, x_ND)

    def parallelize(self, parallel_dims) -> None:
        """Parallelize expert weights, then wire EP/TP meshes on the dispatcher
        so dispatch/combine see the right meshes at runtime."""
        super().parallelize(parallel_dims)
        # TODO(@pianpwk): With spmd_types and set_current_mesh, replace wire_meshes
        # with current_mesh calls inside AllToAllTokenDispatcher and
        # DeepEPTokenDispatcher.
        self.token_dispatcher.wire_meshes(
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
        )


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
        self, scores_BSE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment."""
        leading_shape = scores_BSE.shape[:-1]
        n_tokens = scores_BSE[..., 0].numel()
        selected_experts_indices_BSK = (
            torch.arange(
                n_tokens * self.top_k, device=scores_BSE.device, dtype=torch.int64
            ).reshape(*leading_shape, self.top_k)
            % self.num_experts
        )
        top_scores_BSK = scores_BSE.gather(
            dim=-1, index=selected_experts_indices_BSK
        )
        return selected_experts_indices_BSK, top_scores_BSK

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice_BSE: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
        and set expert scores in non-selected groups as -inf.
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
        scores_grouped = scores_for_choice_BSE.unflatten(
            -1, (self.num_expert_groups, experts_per_group)
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(-1, group_idx, False)  # False = selected groups (keep)
        scores_for_choice_BSE = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).flatten(-2)

        return scores_for_choice_BSE

    def forward(
        self, x_BSD: torch.Tensor, expert_bias_E: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_BSD: Input ``(B, S, D)``.
            expert_bias_E: Optional load-balancing bias ``(E,)``.

        Returns:
            top_scores_BSK: Routing scores ``(B, S, K)``.
            selected_experts_indices_BSK: Expert indices ``(B, S, K)``.
            num_tokens_per_expert_E: Token counts per expert ``(E,)``.
        """
        with torch.autocast(device_type=x_BSD.device.type, dtype=torch.float32):
            scores_BSE = self.gate(x_BSD)

        if self.score_func == "sigmoid":
            scores_BSE = torch.sigmoid(scores_BSE)
        elif self.score_func == "softmax":
            scores_BSE = F.softmax(scores_BSE, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice_BSE = (
            scores_BSE if expert_bias_E is None else scores_BSE + expert_bias_E
        )
        if self.num_expert_groups is not None:
            scores_for_choice_BSE = self._get_node_limited_routing_scores(
                scores_for_choice_BSE
            )
        _, selected_experts_indices_BSK = torch.topk(
            scores_for_choice_BSE, k=self.top_k, dim=-1, sorted=False
        )

        # expert_bias is only used for routing; gating values come from
        # the original scores.
        top_scores_BSK = scores_BSE.gather(
            dim=-1, index=selected_experts_indices_BSK
        )

        if self._debug_force_load_balance:
            (
                selected_experts_indices_BSK,
                top_scores_BSK,
            ) = self._debug_force_load_balance_routing(scores_BSE)

        if self.route_norm:
            denominator = top_scores_BSK.sum(dim=-1, keepdim=True) + 1e-20
            top_scores_BSK = top_scores_BSK / denominator
        top_scores_BSK = top_scores_BSK * self.route_scale

        num_tokens_per_expert_E = torch.histc(
            selected_experts_indices_BSK.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores_BSK, selected_experts_indices_BSK, num_tokens_per_expert_E


class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments (stays on DTensor)
    2. GroupedExperts.forward() converts DTensor to local, then handles:
       a. dispatch (TokenDispatcher) — reorder tokens by expert assignment.
          With EP, also performs all-to-all communication to send tokens
          to expert-owning ranks.
       b. expert computation (local tensors)
       c. combine (TokenDispatcher) — reverse the dispatch reordering.
          - LocalTokenDispatcher (no EP): scatter_add only.
          - AllToAll: all-to-all communication, then scatter_add.
          - DeepEP: async combine_tokens (sync deferred to step 4 when
            sp_size == 1; forced inside combine when sp_size > 1).
          - HybridEP: synchronous combine_tokens.
    3. Shared experts run on DTensor. Overlaps with DeepEP async combine
       when sp_size == 1; no overlap otherwise.
    4. Routed and shared expert outputs are summed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        experts: GroupedExperts.Config
        router: TokenChoiceTopKRouter.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.experts = config.experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )

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

    def forward(self, x_BSD: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_BSD: Input ``(B, S, D)``.

        Returns:
            Output ``(B, S, D)``.

        Under TP, the MoE wrapper's ``sharding_config`` (set by
        ``set_moe_sharding_config``) handles input/output redistribution:
        input is redistributed from sp_layout to desired_input_layouts;
        output (Partial) is redistributed to sp_layout. MoE.forward()
        operates on DTensors — the DTensor→local conversion happens at
        the GroupedExperts boundary.
        """
        bs, slen, dim = x_BSD.shape

        (
            top_scores_BSK,
            selected_experts_indices_BSK,
            num_tokens_per_expert_E,
        ) = self.router(x_BSD, self.expert_bias)

        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert_E)

        out_ND = self.experts(
            x_BSD, top_scores_BSK, selected_experts_indices_BSK
        )

        shared_out_BSD = (
            self.shared_experts(x_BSD) if self.shared_experts is not None else None
        )

        if (
            isinstance(self.experts.token_dispatcher, DeepEPTokenDispatcher)
            and self.experts.token_dispatcher.sp_size == 1
        ):
            from torchtitan.distributed.deepep.deepep import sync_combine

            sync_combine()

        out_BSD = out_ND.reshape(bs, slen, dim)
        if shared_out_BSD is not None:
            out_BSD = out_BSD + shared_out_BSD
        return out_BSD

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
