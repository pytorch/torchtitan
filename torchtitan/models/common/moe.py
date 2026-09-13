# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn

from torchtitan.distributed.spmd_types import (
    maybe_set_sparse_mesh,
    spmd_local_context,
    spmd_mesh_size,
    spmd_sparse_mesh,
)
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import RouterGateLinear
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
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                # if no EP, convert to V for grouped_mm, which would otherwise see
                # x:R, w1:V, offsets:P in local SPMD typechecking.
                # spmd.P is not currently allowed to mix with spmd.V.
                # TODO(pianpwk): likely relax this in spmd_types.
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        h_RF = F.silu(
            self._grouped_mm(A=x_RD.bfloat16(), weight_EOI=self.w1_EFD, offs=offsets_E)
        )
        h_RF = h_RF * self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=self.w3_EFD, offs=offsets_E
        )
        return self._grouped_mm(A=h_RF, weight_EOI=self.w2_EDF, offs=offsets_E).type_as(
            x_RD
        )

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
    """Local SPMD region composing token_dispatcher and inner_experts
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

        When parallelized, ``local_spmd`` (from ``sharding_config``) establishes
        the local SPMD types for the forward body.
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
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: RouterGateLinear.Config
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        aux_loss: AuxLoss.Config | None = None
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None
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

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
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
            routing_map_TE: One-hot boolean routing map ``(T, E)``.
        """
        scores_TE = self.gate(x_TD)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion.
        # RouterGateLinear returns scores_TE in FP32.
        if self.score_func == "sigmoid":
            scores_TE = torch.sigmoid(scores_TE)
        elif self.score_func == "softmax":
            scores_TE = F.softmax(scores_TE, dim=-1)
        elif self.score_func == "sqrtsoftplus":
            scores_TE = F.softplus(scores_TE).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self._debug_force_load_balance:
            topk_expert_ids_TK, topk_scores_TK = self._debug_force_load_balance_routing(
                scores_TE
            )
        else:
            # Routing choices must remain identical between forward and replay.
            topk_expert_ids_TK = remat.region(
                self._select_experts,
                "routing_decision",
                recompute=False,
            )(scores_TE, expert_bias_E, **router_kwargs)
            remat.recompute_needs_tensor(topk_expert_ids_TK)

            # The expert bias is only used for routing. The gating value is
            # still derived from the original scores.
            topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)

        if self.route_norm:
            denominator = topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_TK = topk_scores_TK / denominator
        topk_scores_TK = topk_scores_TK * self.route_scale

        # Build a one-hot boolean routing map (T, E) marking the experts each
        # token is routed to. Under TP/SP the router outputs are sharded on the
        # token dimension; scatter_ writes along the replicated expert
        # dimension and therefore needs no redistribution.
        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids_TK,
            True,
        )

        # Auxiliary load-balance loss (DeepSeek-V3 Sec 2.1.2 Eqs 17-20).
        # The gradient is injected into topk_scores_TK on backward; the loss
        # itself keeps its forward-side metric accumulation from being re-run
        # by activation checkpointing (see ``AuxLoss.inject``).  The
        # routing map is passed in so the loss counts exactly the tokens this
        # router counted: once the router masks padding positions out of the
        # map, the loss and its token count follow without further changes.
        if self.training and self.aux_loss is not None:
            topk_scores_TK = self.aux_loss(
                scores_TE,
                routing_map_TE,
                carrier=topk_scores_TK,
            )

        return (
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        )


class MicrobatchWiseLoadBalanceLoss(AuxLoss):
    """Per-forward MoE load-balance gradient (DeepSeek-V3 Sec 2.1.2 Eqs 17-20).

    The balancing unit is one forward's folded token stream (a DP-local
    microbatch).  Global (corpus-level) balance is left to the
    auxiliary-loss-free bias path (``expert_bias_E``); this loss only
    discourages extreme load imbalance within individual forwards (samples),
    per the DeepSeek-V3 design (Sec 2.1.2, "Complementary Sequence-Wise
    Auxiliary Loss").

    With ``E`` experts, top-``K`` selection and ``T`` tokens per forward:

    Eq. 18: ``f_i = (E / (K T)) * sum_t 1[token t routes to expert i]``
    Eq. 19: ``p_i = (1 / T) * sum_t s'_t,i``,
            where ``s'_t,i = s_t,i / sum_j s_t,j`` is the per-token
            normalized score.
    Eq. 17: ``L_bal = sum_i f_i * p_i``

    The returned value is ``T * L_bal`` (token-mode): Eqs 17-20 define a
    per-token-normalized value, while ``AuxLoss`` scales every auxiliary
    loss by ``1 / global_valid_tokens`` (the step's valid-token count), so the
    sum-type form keeps the injected weight at ``coeff * L_bal``.

    The counts (Eq. 18) and normalized-score sums (Eq. 19) are sums over the
    folded token dim, hence Partial over the mesh axes that shard it (CP
    always, plus TP under EP).  They are all-reduced to Invariant before
    the formula, so every rank computes the same per-forward loss.  The
    one-hot counts are non-differentiable: the gradient reaches the router
    only through the normalized-score sums and the top-k score carrier.
    ``T`` is the forward's token count: the code evaluates Eq. 18 in the
    T-free form ``f_i = E * counts_i / sum_j counts_j``, which equals
    ``(E / (K T)) * counts_i`` because each token contributes K entries, so
    ``sum_j counts_j = K T``.  That needs no shape or mesh-degree assumption
    and follows any masking the router applies to the routing map.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        """Same fields as ``AuxLoss.Config``; this loss adds no knobs.

        A distinct Config is required even without new fields: ``Config.build()``
        constructs the class that owns the config (``__init_subclass__`` sets
        ``_owner``), so a router configured with ``AuxLoss.Config`` would
        build a plain ``AuxLoss``, which has no ``forward``.
        """

    def _reduce_token_partials(
        self, partial_E: torch.Tensor, axes: tuple[str, ...]
    ) -> torch.Tensor:
        """Partial -> Invariant all-reduce over the token-partition axes.

        Axes are passed by name, so spmd_types resolves them against the
        ambient mesh and no DeviceMesh escapes into model code; an inactive
        axis is skipped rather than run as a no-op collective.  ``P -> I`` is
        an all-reduce in forward with an identity backward: the reduced sums,
        and hence the loss and its gradient, are identical on every rank of
        the reduction group.
        """
        for axis in axes:
            if spmd_mesh_size(axis) == 1:
                # No mesh context or a size-1 axis: nothing shards the tokens.
                continue
            partial_E = spmd.redistribute(
                partial_E,
                axis,
                src=spmd.Partial,
                dst=spmd.Invariant,
                backward_options={"op_dtype": partial_E.dtype},
            )
        return partial_E

    def forward(
        self,
        scores_TE: torch.Tensor,
        routing_map_TE: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-forward balance loss and inject its gradient.

        Args:
            scores_TE: Router scores ``(T, E)`` for the forward's tokens.
            routing_map_TE: One-hot routing map ``(T, E)`` for the same tokens,
                as counted by the router.
            carrier: Tensor whose backward path carries the injected
                gradient (the router's top-k scores).

        Returns:
            ``carrier`` unchanged (identity forward).
        """
        # Mark DP local for the counts arithmetic: each DP rank owns an
        # independent token stream, so DP must not be reduced; only the
        # global axes that shard the stream (CP, TP under EP) are.
        with spmd_local_context("dp"):
            E = scores_TE.size(-1)
            # Axes that shard the router output's token dim: CP in every
            # layout, TP only under EP, which distributes tokens over TP (the
            # gate computes and emits dense_sequence_parallel_placement
            # whenever EP is on, and tokens_per_expert_E is TP-Partial for the
            # same reason).
            axes = ("cp", "tp") if spmd_sparse_mesh() is not None else ("cp",)

            # Eq. 18: per-expert routing frequency counts_i over the forward's
            # tokens, then f_i = E * counts_i / sum_j counts_j (so
            # sum_i f_i = E).  The latter is the (E / (K T)) form with
            # T = sum_j counts_j / K, so it needs no token count, shape or mesh
            # degree and follows any masking the router applies to the map.
            # The map is cast to float before the reduction: casting a Partial
            # tensor is non-linear and rejected by spmd_types.
            counts_E = self._reduce_token_partials(
                routing_map_TE.to(scores_TE.dtype).sum(dim=0), axes
            )
            f_E = F.normalize(counts_E, p=1, dim=0) * E

            # Eq. 19: p_i = (1/T) sum_t s'_t,i, the per-token L1-normalized
            # scores.  F.normalize's eps clamp only guards an all-zero score
            # row: the scores are non-negative, so the norm is a plain sum.
            probs_TE = F.normalize(scores_TE, p=1, dim=-1)
            p_E = self._reduce_token_partials(probs_TE.sum(dim=0), axes)

            # Eq. 17: L_bal = sum_i f_i * p_i
            loss = (f_E * p_E).sum()
            return self.inject(loss, carrier=carrier)


class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments.
    2. RoutedExperts.forward() enters a local SPMD region, then handles:
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
        output is redistributed to sp_layout. GroupedExperts operates in a
        local SPMD region. When EP internally
        sequence-shards tokens across TP, the caller must provide a TP-divisible
        token count.
        """
        # topk scores and expert IDs have shape (T, K); the routing map (T, E)
        # marks the experts each token is routed to (built inside the router).
        (
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        ) = self.router(x_TD, self.expert_bias_E, **router_kwargs)
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
