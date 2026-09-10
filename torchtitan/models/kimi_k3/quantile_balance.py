# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch_remat as remat
from torch.distributed.tensor import DTensor

from torchtitan.components.optimizer.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter
from torchtitan.protocols.module import Module

# Shape suffixes:
# T = tokens, D = model dimension, E = experts, K = selected experts,
# B = histogram bins.


class QuantileBalancedTopKRouter(TokenChoiceTopKRouter):
    """Top-k router that uses a biased Top-(k+1) cutoff during training."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        num_bins: int = 1000

    def __init__(self, config: Config):
        super().__init__(config)
        if self.score_func != "sigmoid":
            raise ValueError("Quantile balancing requires sigmoid router scores.")
        if self.num_expert_groups is not None:
            raise ValueError(
                "Quantile balancing does not support group-limited routing."
            )
        if self._debug_force_load_balance:
            raise ValueError(
                "Quantile balancing does not support forced debug load balancing."
            )
        self.quantile_balancer = QuantileBalancer.Config(
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_bins=config.num_bins,
        ).build()

    def _select_experts_and_cutoff(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        biased_scores_TE = scores_TE + expert_bias_E
        topk_plus_one_scores, topk_plus_one_expert_ids = torch.topk(
            biased_scores_TE,
            k=self.top_k + 1,
            dim=-1,
            sorted=True,
        )
        return (
            topk_plus_one_expert_ids[:, : self.top_k],
            topk_plus_one_scores[:, self.top_k :],
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens and collect the required-bias histogram in training."""
        if expert_bias_E is None:
            raise ValueError("Quantile balancing requires an expert bias.")
        if not self.training:
            return super().forward(x_TD, expert_bias_E, **router_kwargs)

        scores_TE = torch.sigmoid(self.gate(x_TD))
        topk_expert_ids_TK, cutoff_T1 = remat.region(
            self._select_experts_and_cutoff,
            "routing_decision",
            recompute=False,
        )(scores_TE, expert_bias_E)
        remat.recompute_needs_tensor(topk_expert_ids_TK, cutoff_T1)
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
        if self.route_norm:
            denominator_T1 = topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_TK = topk_scores_TK / denominator_T1
        topk_scores_TK = topk_scores_TK * self.route_scale
        self.quantile_balancer.observe(scores_TE, cutoff_T1, expert_bias_E)
        return topk_scores_TK, topk_expert_ids_TK, scores_TE


class QuantileBalancer(Module):
    """Accumulate and recover Kimi K3's histogram quantile bias update.

    For bounded sigmoid scores, required expert biases lie between the current
    minimum bias minus one and maximum bias plus one. Each training micro-batch
    is accumulated into uniform bins over that interval. The optimizer hook
    pools those counts globally before recovering the target quantile.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        top_k: int
        num_bins: int = 1000

    def __init__(self, config: Config):
        super().__init__()
        if not 0 < config.top_k < config.num_experts:
            raise ValueError("top_k must be between zero and num_experts.")
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.num_bins = config.num_bins
        self.register_buffer(
            "required_bias_histogram_EB",
            torch.zeros(config.num_experts, config.num_bins, dtype=torch.int32),
            persistent=False,
        )

    @staticmethod
    def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def observe(
        self,
        scores_TE: torch.Tensor,
        cutoff_T1: torch.Tensor,
        expert_bias_E: torch.Tensor,
    ) -> None:
        """Accumulate required-bias histograms for one local micro-batch."""
        if not self.training:
            return

        with torch.no_grad():
            local_scores_TE = self._local_tensor(scores_TE)
            local_cutoff_T1 = self._local_tensor(cutoff_T1)
            local_expert_bias_E = self._local_tensor(expert_bias_E)

            lower_bound = local_expert_bias_E.min() - 1.0
            bin_width = (
                local_expert_bias_E.max() - local_expert_bias_E.min() + 2.0
            ) / self.num_bins
            required_bias_TE = local_cutoff_T1 - local_scores_TE
            bin_indices_TE = torch.floor(
                (required_bias_TE - lower_bound) / bin_width
            ).to(torch.int64)
            bin_indices_ET = bin_indices_TE.clamp_(0, self.num_bins - 1).transpose(0, 1)
            microbatch_histogram_EB = torch.zeros_like(self.required_bias_histogram_EB)
            microbatch_histogram_EB.scatter_add_(
                1,
                bin_indices_ET,
                torch.ones_like(
                    bin_indices_ET,
                    dtype=microbatch_histogram_EB.dtype,
                ),
            )
            self.required_bias_histogram_EB.add_(microbatch_histogram_EB)

    def estimate_expert_bias(
        self,
        expert_bias_E: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the next mean-centered expert bias from the histogram."""
        local_expert_bias_E = self._local_tensor(expert_bias_E)
        histogram_EB = self.required_bias_histogram_EB

        counts_E = histogram_EB.sum(dim=-1, dtype=torch.int64)
        target_count_E = counts_E.float() * (self.top_k / self.num_experts)
        cumulative_counts_EB = histogram_EB.cumsum(dim=-1, dtype=torch.int64)
        target_rank_E = target_count_E.ceil().to(torch.int64)
        target_bin_E = (cumulative_counts_EB < target_rank_E.unsqueeze(-1)).sum(dim=-1)

        target_bin_E1 = target_bin_E.unsqueeze(-1)
        counts_in_bin_E = histogram_EB.gather(-1, target_bin_E1).squeeze(-1)
        counts_before_E = (
            cumulative_counts_EB.gather(-1, target_bin_E1).squeeze(-1) - counts_in_bin_E
        )
        fraction_E = (
            target_count_E - counts_before_E.float()
        ) / counts_in_bin_E.float()

        bin_width = (
            local_expert_bias_E.max() - local_expert_bias_E.min() + 2.0
        ) / self.num_bins
        quantile_position_E = target_bin_E.float() + fraction_E
        return (quantile_position_E - quantile_position_E.mean()) * bin_width

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.required_bias_histogram_EB.device
        with torch.device(buffer_device):
            self.required_bias_histogram_EB = torch.zeros(
                self.num_experts,
                self.num_bins,
                dtype=torch.int32,
            )


def register_moe_quantile_balancing_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Update Kimi K3 expert biases from globally reduced histograms."""
    moe_layers: list[tuple[MoE, QuantileBalancedTopKRouter]] = []
    for model_part in model_parts:
        for module in model_part.modules():
            if not isinstance(module, MoE):
                continue
            if not isinstance(module.router, QuantileBalancedTopKRouter):
                raise ValueError("All Kimi K3 MoE layers must use quantile balancing.")
            moe_layers.append((module, module.router))

    if not moe_layers:
        return

    def _all_reduce_histograms(group) -> None:
        handles = [
            torch.distributed.all_reduce(
                router.quantile_balancer.required_bias_histogram_EB,
                group=group,
                op=torch.distributed.ReduceOp.SUM,
                async_op=True,
            )
            for _moe, router in moe_layers
        ]
        for handle in handles:
            handle.wait()

    @torch.no_grad()
    def _update_expert_bias() -> None:
        # With EP, the router is token-sharded on the dense TP axis even when
        # model-wide sequence parallelism is disabled.
        if parallel_dims.ep_enabled and parallel_dims.tp > 1:
            _all_reduce_histograms(
                parallel_dims.get_dense_tp_mesh().get_group(),
            )
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        if loss_mesh is not None:
            _all_reduce_histograms(loss_mesh.get_group())

        for moe, router in moe_layers:
            expert_bias_E = moe.expert_bias_E
            assert expert_bias_E is not None
            quantile_balancer = router.quantile_balancer
            next_expert_bias_E = quantile_balancer.estimate_expert_bias(expert_bias_E)
            if isinstance(expert_bias_E, DTensor):
                expert_bias_E = expert_bias_E.to_local()
            expert_bias_E.copy_(next_expert_bias_E)
            quantile_balancer.required_bias_histogram_EB.zero_()

    optimizers.register_step_pre_hook(lambda *args, **kwargs: _update_expert_bias())
