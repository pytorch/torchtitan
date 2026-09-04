# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh, spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

from .token_dispatcher import LocalTokenDispatcher

# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   T = num tokens, D = model dimension,
#   F = hidden (FFN intermediate) dimension, E = num experts,
#   e = num local experts (E / EP, used in token dispatcher for
#       per-local-expert token counts after EP dispatch /_permute),
#   K = top-k, N = routed tokens (T*K),
#   R = routed tokens assigned to local experts,
#   O = expert output features, I = expert input features
#       (roles, not model dims: the _grouped_mm seam takes the expert
#        weight in its stored (E, O, I) orientation, which is (E, F, D)
#        for the up/gate projections and (E, D, F) for the down one)


class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w1_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2_EDF = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine.

        Shape suffixes here describe logical grouped-mm inputs, not physical
        sharding. Under EP, E may be a local shard of experts; under TP,
        expert weights shard hidden dimensions instead; under SP, R may be a
        local token shard. Keep logical capital suffixes here to avoid encoding
        a specific parallel layout in these local tensor names.
        """
        if isinstance(self.w1_EFD, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if (
            get_spmd_backend() == "spmd_types"
            and spmd.is_type_checking()
            and spmd_mesh_size("ep") == 1
        ):
            for axis in ("dp", "cp"):
                # if no EP, convert to V for grouped_mm, which would otherwise see
                # x:R, w1:V, offsets:P in local SPMD typechecking.
                # spmd.P is not currently allowed to mix with spmd.V.
                # TODO(pianpwk): likely relax this in spmd_types.
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        h_RF = F.silu(
            self._grouped_mm(A=x_RD.bfloat16(), weight_EOI=w1_EFD, offs=offsets_E)
        )
        h_RF = h_RF * self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=w3_EFD, offs=offsets_E
        )
        return self._grouped_mm(A=h_RF, weight_EOI=w2_EDF, offs=offsets_E).type_as(x_RD)

    def _grouped_mm(
        self, *, A: torch.Tensor, weight_EOI: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        """Grouped matmul of ``A @ weight_EOI.transpose(-2, -1)``.

        ``weight_EOI`` is the grouped expert weight in its stored
        ``(experts, out_features, in_features)`` orientation; the transpose to
        the grouped-GEMM right operand happens here. Overridable seam for
        low-precision variants (e.g. the MXFP8 converter swaps this for a
        scaled grouped GEMM). Variants receive the weight rather than its
        transpose because a quantized representation may be owned by the
        weight's FSDP unshard lifetime and is keyed off the stored orientation.
        Keeping the op here -- rather than behind a tensor-subclass
        ``__torch_function__`` -- means it is captured by FX tracers such as
        graph_trainer's make_fx path.
        """
        return torch._grouped_mm(A, weight_EOI.bfloat16().transpose(-2, -1), offs=offs)


class RoutedExperts(Module):
    """Routed-expert ``local_map`` region: composes token_dispatcher + inner_experts
    as sibling nodes so each can be overridden independently."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        inner_experts: GroupedExperts.Config
        token_dispatcher: LocalTokenDispatcher.Config

    def __init__(self, config: Config):
        super().__init__()
        self.inner_experts = config.inner_experts.build()
        self.token_dispatcher = config.token_dispatcher.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        When parallelized, ``local_map`` (from ``sharding_config``) handles
        DTensor→local conversion on entry and local→DTensor(Partial) wrapping
        on exit. The forward body operates on plain local tensors.
        """
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            routed_output_RD = self.inner_experts(
                routed_input_RD, num_global_tokens_per_local_expert_e
            )
        out_TD = self.token_dispatcher.combine(
            routed_output_RD,
            metadata,
            x_TD,
        )
        return out_TD

    def parallelize(self, parallel_dims) -> None:
        """Parallelize the grouped experts, then wire the EP mesh on the
        dispatcher so dispatch/combine see the right mesh at runtime."""
        super().parallelize(parallel_dims)
        # TODO(@pianpwk): With spmd_types and set_current_spmd_mesh, replace wire_meshes
        # with current_spmd_mesh calls inside AllToAllTokenDispatcher and
        # DeepEPTokenDispatcher.
        self.token_dispatcher.wire_meshes(
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
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
        score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sigmoid"
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
        self, scores_TE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns expert IDs and scores with shape ``(T, K)``.
        """
        num_tokens = scores_TE.shape[0]
        # Round-robin indices with exact balance
        topk_expert_ids_TK = (
            torch.arange(
                num_tokens * self.top_k,
                device=scores_TE.device,
                dtype=torch.int64,
            ).reshape(num_tokens, self.top_k)
            % self.num_experts
        )
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
        return topk_expert_ids_TK, topk_scores_TK

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice_TE: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
        and set expert scores in non-selected groups as -inf.

        Args:
            scores_for_choice_TE: Router scores with expert_bias, shape ``(T, E)``.

        Returns:
            Router scores with shape ``(T, E)``.
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
        scores_grouped = scores_for_choice_TE.unflatten(
            -1, (self.num_expert_groups, experts_per_group)
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(-1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice_TE = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).flatten(-2)

        return scores_for_choice_TE

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        if self.num_expert_groups is not None:
            scores_for_choice_TE = self._get_node_limited_routing_scores(
                scores_for_choice_TE
            )
        return torch.topk(
            scores_for_choice_TE, k=self.top_k, dim=-1, sorted=False
        ).indices

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_TD: Input ``(T, D)``.
            expert_bias_E: Optional load-balancing bias ``(E,)``.

        Returns:
            topk_scores_TK: Routing scores ``(T, K)``.
            topk_expert_ids_TK: Expert indices ``(T, K)``.
            scores_TE: Full routing scores ``(T, E)``.
        """
        # Compute gate in float32 to help stability of expert load balancing.
        with torch.autocast(device_type=x_TD.device.type, dtype=torch.float32):
            scores_TE = self.gate(x_TD)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion.
        # scores_TE is already float32 from the autocast above.
        if self.score_func == "sigmoid":
            scores_TE = torch.sigmoid(scores_TE)
        elif self.score_func == "softmax":
            scores_TE = F.softmax(scores_TE, dim=-1)
        elif self.score_func == "sqrtsoftplus":
            scores_TE = F.softplus(scores_TE).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        topk_expert_ids_TK = self._select_experts(
            scores_TE, expert_bias_E, **router_kwargs
        )

        # NOTE: The expert_bias is only used for routing. The gating value
        #       topk_scores_TK is still derived from the original scores.
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                topk_expert_ids_TK,
                topk_scores_TK,
            ) = self._debug_force_load_balance_routing(scores_TE)

        if self.route_norm:
            denominator = topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_TK = topk_scores_TK / denominator
        topk_scores_TK = topk_scores_TK * self.route_scale

        return (
            topk_scores_TK,
            topk_expert_ids_TK,
            scores_TE,
        )


class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments (stays on DTensor)
    2. RoutedExperts.forward() converts DTensor to local, then handles:
       a. dispatch (TokenDispatcher) — reorder tokens by expert assignment.
          With EP, also performs all-to-all communication to send tokens
          to expert-owning ranks.
       b. expert computation (GroupedExperts, local tensors)
       c. combine (TokenDispatcher) — reverse the dispatch reordering.
          - LocalTokenDispatcher (no EP): scatter_add only.
          - AllToAll: all-to-all communication, then scatter_add.
          - DeepEP: combine_tokens followed by backend synchronization.
          - HybridEP: synchronous combine_tokens.
    3. Shared experts compute their output.
    4. Routed and shared expert outputs are summed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        routed_experts: RoutedExperts.Config
        router: TokenChoiceTopKRouter.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.routed_experts = config.routed_experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert_E is accumulated in the model forward pass.
        #       expert_bias_E is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias_E",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias_E = None
        # tokens_per_expert_E will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x_TD: torch.Tensor, **router_kwargs) -> torch.Tensor:
        """
        Args:
            x_TD: Input ``(T, D)``.

        Returns:
            Output ``(T, D)``.

        Under TP, the MoE wrapper's ``sharding_config`` (set by
        ``set_moe_sharding_config``) handles input/output redistribution:
        input is redistributed from sp_layout to desired_input_layouts;
        output is redistributed to sp_layout. MoE.forward() operates on
        DTensors; the DTensor->local conversion happens at the GroupedExperts
        boundary. GroupedExperts operates on local tensors. When EP internally
        sequence-shards tokens across TP, the caller must provide a TP-divisible
        token count.
        """
        # topk scores and expert IDs have shape (T, K); scores have shape (T, E).
        (
            topk_scores_TK,
            topk_expert_ids_TK,
            scores_TE,
        ) = self.router(x_TD, self.expert_bias_E, **router_kwargs)

        # Build a one-hot routing map (T, E) marking the experts each token
        # is routed to. Under TP/SP the router outputs are DTensors sharded on
        # the token dim; scatter_ writes along the (replicated) expert dim, so
        # DTensor runs it as a local op with no redistribution.
        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids_TK,
            True,
        )
        num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0)

        # tokens_per_expert_E will be used to update the expert bias for load balancing,
        # and also to count the expert usage.
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert_E --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_TD = self.routed_experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )

        shared_out_TD = (
            self.shared_experts(x_TD) if self.shared_experts is not None else None
        )

        if shared_out_TD is not None:
            out_TD = out_TD + shared_out_TD
        return out_TD

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            # After ``to_empty()``, the existing buffer records the target device.
            # Reinitialize MoE counters there when no explicit buffer device is passed.
            buffer_device = self.tokens_per_expert_E.device

        with torch.device(buffer_device):
            self.tokens_per_expert_E = torch.zeros(
                self.routed_experts.inner_experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias_E = torch.zeros(
                    self.routed_experts.inner_experts.num_experts, dtype=torch.float32
                )
