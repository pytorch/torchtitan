# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.module import Module

from .token_dispatcher import DeepEPTokenDispatcher, LocalTokenDispatcher

# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length, D = model dimension,
#   F = hidden (FFN intermediate) dimension, E = num experts,
#   e = num local experts (E / EP, used in token dispatcher for
#       per-local-expert token counts after EP dispatch /_permute),
#   K = top-k, T = num tokens (B*L flattened),
#   N = routed tokens (T*K), R = routed tokens assigned to local experts


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
        self.w1_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2_EDF = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.token_dispatcher = config.token_dispatcher.build()

    def _experts_forward(
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
            # pyrefly: ignore [missing-attribute]
            w2_EDF = self.w2_EDF.to_local()
            # pyrefly: ignore [missing-attribute]
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)

        h_RF = F.silu(
            torch._grouped_mm(
                x_RD.bfloat16(),
                w1_EFD.bfloat16().transpose(-2, -1),
                offs=offsets_E,
            )
        )
        h_RF = h_RF * torch._grouped_mm(
            x_RD.bfloat16(),
            w3_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        return torch._grouped_mm(
            h_RF, w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E
        ).type_as(x_RD)

    def forward(
        self,
        x_BLD: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        When parallelized, ``local_map`` (from ``sharding_config``) handles
        DTensor→local conversion on entry and local→DTensor(Partial) wrapping
        on exit. The forward body operates on plain local tensors.
        """
        B, L, D = x_BLD.shape
        K = topk_scores_BLK.size(-1)
        T = B * L
        x_TD = x_BLD.view(T, D)
        topk_scores_TK = topk_scores_BLK.view(T, K)
        topk_expert_ids_TK = topk_expert_ids_BLK.view(T, K)
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )
        routed_output_RD = self._experts_forward(
            routed_input_RD, num_global_tokens_per_local_expert_e
        )
        return self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)

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
        self, scores_BLE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (topk_expert_ids_BLK ``(B, L, K)`` LongTensor, topk_scores_BLK ``(B, L, K)`` FloatTensor).
        """
        bs, slen, _ = scores_BLE.shape
        # Round-robin indices with exact balance
        topk_expert_ids_BLK = (
            torch.arange(
                bs * slen * self.top_k, device=scores_BLE.device, dtype=torch.int64
            ).reshape(bs, slen, self.top_k)
            % self.num_experts
        )
        topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)
        return topk_expert_ids_BLK, topk_scores_BLK

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice_BLE: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
        and set expert scores in non-selected groups as -inf.

        Args:
            scores_for_choice_BLE: Router scores with expert_bias (if any), shape ``(B, L, E)``.

        Returns:
            scores_for_choice_BLE: shape ``(B, L, E)``.
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
        scores_grouped = scores_for_choice_BLE.unflatten(
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
        scores_for_choice_BLE = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).flatten(-2)

        return scores_for_choice_BLE

    def forward(
        self, x_BLD: torch.Tensor, expert_bias_E: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_BLD: Input ``(B, L, D)``.
            expert_bias_E: Optional load-balancing bias ``(E,)``.

        Returns:
            topk_scores_BLK: Routing scores ``(B, L, K)``.
            topk_expert_ids_BLK: Expert indices ``(B, L, K)``.
            scores_BLE: Full routing scores ``(B, L, E)``.
        """
        # Compute gate in float32 to help stability of expert load balancing.
        with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
            scores_BLE = self.gate(x_BLD)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion.
        # scores_BLE is already float32 from the autocast above.
        if self.score_func == "sigmoid":
            scores_BLE = torch.sigmoid(scores_BLE)
        elif self.score_func == "softmax":
            scores_BLE = F.softmax(scores_BLE, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice_BLE = (
            scores_BLE if expert_bias_E is None else scores_BLE + expert_bias_E
        )
        # Apply node-limited routing if configured
        if self.num_expert_groups is not None:
            scores_for_choice_BLE = self._get_node_limited_routing_scores(
                scores_for_choice_BLE
            )
        _, topk_expert_ids_BLK = torch.topk(
            scores_for_choice_BLE, k=self.top_k, dim=-1, sorted=False
        )

        # NOTE: The expert_bias is only used for routing. The gating value
        #       topk_scores_BLK is still derived from the original scores.
        topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                topk_expert_ids_BLK,
                topk_scores_BLK,
            ) = self._debug_force_load_balance_routing(scores_BLE)

        if self.route_norm:
            denominator = topk_scores_BLK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_BLK = topk_scores_BLK / denominator
        topk_scores_BLK = topk_scores_BLK * self.route_scale

        return (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
        )


def _sequence_wise_load_balance_loss(
    scores_BLE: torch.Tensor,
    topk_expert_ids_BLK: torch.Tensor,
    top_k: int,
    aux_loss_coeff: float,
    *,
    cp_mesh: DeviceMesh | None = None,
    tp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """Sequence-wise auxiliary load-balance loss (DeepSeek-V3 Eqs 17-20).

    For each sequence, compute f_i * p_i across that sequence's expert
    probability scores, then average the resulting load-balance losses over the
    batch.

    With sequence partitioning, counts_BE starts as per-batch,
    per-sequence-shard expert counts and is all-reduced across sequence-shard
    groups (CP, and TP when EP enables sequence parallelism) to get full-sequence
    expert counts.
    """
    B, L, num_experts = scores_BLE.shape
    denom = scores_BLE.sum(dim=-1, keepdim=True) + 1e-20
    probs_BLE = scores_BLE / denom
    prob_sums_BE = probs_BLE.sum(dim=1)
    indices_BN = topk_expert_ids_BLK.reshape(B, -1)
    offset = torch.arange(B, device=indices_BN.device).unsqueeze(1) * num_experts
    flat_indices = (indices_BN + offset).reshape(-1)
    counts = torch.bincount(flat_indices.long(), minlength=B * num_experts)
    counts_BE = counts.reshape(B, num_experts).to(dtype=scores_BLE.dtype)

    counts_BE = _all_reduce_expert_counts(
        counts_BE,
        reduction="sequence",
        dp_mesh=None,
        cp_mesh=cp_mesh,
        tp_mesh=tp_mesh,
    )
    num_tokens_B = counts_BE.sum(dim=1) / top_k

    p_BE = prob_sums_BE / num_tokens_B.unsqueeze(1)
    f_BE = counts_BE * (num_experts / (top_k * num_tokens_B.unsqueeze(1)))
    return (f_BE * p_BE).sum(dim=1).mean() * aux_loss_coeff


def _batch_wise_load_balance_loss(
    scores_BLE: torch.Tensor,
    topk_expert_ids_BLK: torch.Tensor,
    top_k: int,
    aux_loss_coeff: float,
    *,
    dp_mesh: DeviceMesh | None = None,
    cp_mesh: DeviceMesh | None = None,
    tp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """Batch-wise auxiliary load-balance loss.

    Computes f_i * p_i using expert frequency across the global batch. Counts
    start as an [E] per-expert count for the local batch and sequence shard, then
    are all-reduced across DP and sequence-shard groups (CP, and TP when EP
    enables sequence parallelism) to get global-batch expert counts.
    """
    B, L, num_experts = scores_BLE.shape
    num_tokens_per_expert = torch.histc(
        topk_expert_ids_BLK.reshape(-1).float(),
        bins=num_experts,
        min=0,
        max=num_experts,
    )
    probs_BLE = scores_BLE / (scores_BLE.sum(dim=-1, keepdim=True) + 1e-20)
    prob_sums_E = probs_BLE.sum(dim=(0, 1))

    num_tokens_per_expert = num_tokens_per_expert.to(scores_BLE.dtype)
    global_num_tokens_per_expert = _all_reduce_expert_counts(
        num_tokens_per_expert,
        reduction="batch",
        dp_mesh=dp_mesh,
        cp_mesh=cp_mesh,
        tp_mesh=tp_mesh,
    )
    global_tokens = global_num_tokens_per_expert.sum() / top_k

    # DP contributes through summed gradients and scaled_coeff, so p_i uses the
    # non-DP token count for this local batch contribution.
    dp_degree = dp_mesh.size() if dp_mesh is not None else 1
    local_non_dp_tokens = global_tokens / dp_degree
    p_i = prob_sums_E / local_non_dp_tokens
    f_i = global_num_tokens_per_expert * (
        num_experts / (top_k * global_tokens)
    )
    return (f_i * p_i).sum() * aux_loss_coeff


def _all_reduce_expert_counts(
    counts: torch.Tensor,
    *,
    reduction: Literal["batch", "sequence"],
    dp_mesh: DeviceMesh | None,
    cp_mesh: DeviceMesh | None,
    tp_mesh: DeviceMesh | None,
) -> torch.Tensor:
    """Sum non-differentiable expert counts over token-partition mesh axes.

    The tensor shape is preserved. For batch-wise loss this is ``[E]``; for
    sequence-wise loss this is ``[B, E]`` so the per-sequence objective never
    reduces across the batch dimension.
    """
    assert not counts.requires_grad, "MoE load-balance counts should not require grad"
    if reduction == "batch":
        assert counts.ndim == 1, (
            f"batch-wise expert counts must be 1D, got {counts.shape}"
        )
        reduce_meshes = (tp_mesh, dp_mesh, cp_mesh)
    else:
        assert reduction == "sequence"
        assert counts.ndim == 2, (
            f"sequence-wise expert counts must be 2D, got {counts.shape}"
        )
        assert dp_mesh is None, "sequence-wise expert counts must not reduce over DP"
        reduce_meshes = (tp_mesh, cp_mesh)

    with torch.no_grad():
        for reduce_mesh in reduce_meshes:
            if reduce_mesh is None:
                continue
            dist.all_reduce(
                counts, group=reduce_mesh.get_group(), op=dist.ReduceOp.SUM
            )
    return counts

class _AuxLossAutograd(torch.autograd.Function):
    """Injects an auxiliary loss gradient without changing the forward value."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx: torch.autograd.function.FunctionCtx,
        topk_scores_BLK: torch.Tensor,
        aux_loss: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return topk_scores_BLK

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_topk_scores_BLK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (aux_loss,) = ctx.saved_tensors
        return grad_topk_scores_BLK, torch.ones_like(aux_loss)


class MoELoadBalanceAuxLoss(Module):
    """MoE auxiliary load-balance loss.

    Injects aux loss gradients into router scores without changing the model
    output (PP-safe). Call instance with router outputs to apply.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        type: Literal["sequence_wise", "batch_wise"]
        """Loss variant. sequence_wise (DeepSeek-V3 Eqs 17-20) computes
        per-sequence then averages; batch_wise averages over all tokens."""
        coeff: float
        """Coefficient for the auxiliary load-balance loss. Set moe.aux_loss to
        None to disable aux loss entirely."""
        top_k: int
        """Number of experts per token. Must match the router's top_k."""
        global_batch_size: int | None = None
        """Global batch size to normalize aux loss across all batches/microbatches.
        Torchtitan generally uses sum aggregation & global_bs division.
        Because batch_wise aux loss doesn't commute over microbatches, we ban
        batch_wise + PP usage in any case."""

    def __init__(self, config: Config):
        super().__init__()
        if config.coeff <= 0:
            raise ValueError(
                f"aux_loss.coeff must be > 0, got {config.coeff}. "
                "Set moe.aux_loss to None to disable aux loss."
            )
        if config.type not in ("sequence_wise", "batch_wise"):
            raise ValueError(
                f"Unknown aux loss type '{config.type}', "
                f"expected one of ('sequence_wise', 'batch_wise')"
            )
        self.type = config.type
        self.coeff = config.coeff
        self.global_batch_size = config.global_batch_size
        self.top_k = config.top_k
        self.dp_mesh: DeviceMesh | None = None
        self.cp_mesh: DeviceMesh | None = None
        self.tp_mesh: DeviceMesh | None = None

    def set_meshes(
        self,
        *,
        dp_mesh: DeviceMesh | None,
        cp_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        self.dp_mesh = dp_mesh
        self.cp_mesh = cp_mesh
        self.tp_mesh = tp_mesh

    def forward(
        self,
        topk_scores_BLK: torch.Tensor,
        scores_BLE: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
    ) -> torch.Tensor:
        B = scores_BLE.size(0)
        global_batch_size = self.global_batch_size or B
        scaled_coeff = self.coeff * B / global_batch_size
        if self.type == "sequence_wise":
            aux_loss = _sequence_wise_load_balance_loss(
                scores_BLE,
                topk_expert_ids_BLK,
                self.top_k,
                scaled_coeff,
                cp_mesh=self.cp_mesh,
                tp_mesh=self.tp_mesh,
            )
        else:
            aux_loss = _batch_wise_load_balance_loss(
                scores_BLE,
                topk_expert_ids_BLK,
                self.top_k,
                scaled_coeff,
                dp_mesh=self.dp_mesh,
                cp_mesh=self.cp_mesh,
                tp_mesh=self.tp_mesh,
            )
        return _AuxLossAutograd.apply(topk_scores_BLK, aux_loss)


def configure_moe_aux_loss_reduction(model: nn.Module, parallel_dims) -> None:
    """Wire process meshes used to reduce MoE auxiliary-loss counts.

    The loss functions choose which meshes they need from these handles:
    batch-wise reduces ``[E]`` counts over DP, CP, and TP when applicable.
    Sequence-wise reduces ``[B, E]`` counts over TP (when applicable) and CP
    only, preserving per-sequence counts and avoiding DP.

    In the current MoE sharding, TP only splits router/count tokens when EP is
    enabled: the TP axis doubles as a sequence-parallel split for routing. When
    EP is disabled, router counts are replicated over TP and must not be
    all-reduced on the TP mesh.
    """
    dp_mesh = parallel_dims.get_optional_mesh("batch")
    cp_mesh = parallel_dims.get_optional_mesh("cp")
    tp_mesh = (
        parallel_dims.get_optional_mesh("tp") if parallel_dims.ep_enabled else None
    )
    for module in model.modules():
        if isinstance(module, MoE) and module.aux_loss is not None:
            module.aux_loss.set_meshes(
                dp_mesh=dp_mesh,
                cp_mesh=cp_mesh,
                tp_mesh=tp_mesh,
            )


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

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_BLD: Input ``(B, L, D)``.

        Returns:
            Output ``(B, L, D)``.

        Under TP, the MoE wrapper's ``sharding_config`` (set by
        ``set_moe_sharding_config``) handles input/output redistribution:
        input is redistributed from sp_layout to desired_input_layouts;
        output (Partial) is redistributed to sp_layout. MoE.forward()
        operates on DTensors — the DTensor→local conversion happens at
        the GroupedExperts boundary.
        """
        B, L, D = x_BLD.shape

        # topk_scores_BLK and topk_expert_ids_BLK shape (B, L, K)
        # scores_BLE shape (B, L, E)
        (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
        ) = self.router(x_BLD, self.expert_bias_E)

        if self.training and self.aux_loss is not None:
            topk_scores_BLK = self.aux_loss(
                topk_scores_BLK, scores_BLE, topk_expert_ids_BLK
            )

        # Build routing map with scatter. scatter_ does not support mixed
        # local Tensor / DTensor arguments, so run the scatter on local tensors
        # under local_map when router outputs are DTensors.
        # TODO: Remove this local_map workaround once DTensor sharding
        # propagation supports scatter with mixed Tensor / DTensor arguments.
        def _generate_routing_map(
            scores_BLE: torch.Tensor,
            topk_expert_ids_BLK: torch.Tensor,
        ) -> torch.Tensor:
            return torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
                -1,
                topk_expert_ids_BLK,
                True,
            )

        if isinstance(topk_expert_ids_BLK, DTensor):
            assert isinstance(
                scores_BLE, DTensor
            ), "scores_BLE and topk_expert_ids_BLK should both be DTensors"
            generate_routing_map = local_map(
                _generate_routing_map,
                in_placements=(
                    scores_BLE.placements,
                    topk_expert_ids_BLK.placements,
                ),
                out_placements=(scores_BLE.placements,),
                device_mesh=scores_BLE.device_mesh,
            )
        else:
            generate_routing_map = _generate_routing_map

        routing_map_BLE = generate_routing_map(
            scores_BLE,
            topk_expert_ids_BLK,  # pyrefly: ignore [bad-argument-count]
        )
        num_local_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))

        # tokens_per_expert_E will be used to update the expert bias for load balancing,
        # and also to count the expert usage.
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert_E --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_TD = self.experts(
            x_BLD,
            topk_scores_BLK,
            topk_expert_ids_BLK,
            num_local_tokens_per_expert_E,
        )

        # shared_experts runs in parallel with deepep combine communication.
        shared_out_BLD = (
            self.shared_experts(x_BLD) if self.shared_experts is not None else None
        )

        if (
            isinstance(self.experts.token_dispatcher, DeepEPTokenDispatcher)
            and self.experts.token_dispatcher.sp_size == 1
        ):
            # Sync the combine operation before using routed_output.
            # This inserts a CUDA stream wait, ensuring combine is complete before
            # the subsequent addition or view operations read routed output.
            from torchtitan.distributed.deepep.deepep import sync_combine

            sync_combine()

        out_BLD = out_TD.view(B, L, D)
        if shared_out_BLD is not None:
            out_BLD = out_BLD + shared_out_BLD
        return out_BLD

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            # After ``to_empty()``, the existing buffer records the target device.
            # Reinitialize MoE counters there when no explicit buffer device is passed.
            buffer_device = self.tokens_per_expert_E.device

        with torch.device(buffer_device):
            self.tokens_per_expert_E = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias_E = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
