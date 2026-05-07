# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single_autograd

from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


class HFTokenDispatcher(LocalTokenDispatcher):
    """All-to-all dispatch/combine adapted for HF SparseMoeBlock contract.

    HF MoE blocks call ``experts(hidden_states, top_k_index, top_k_weights)``
    with per-token routing and loop over experts internally. Native
    ``GroupedExperts`` instead consumes ``(routed_input, num_tokens_per_expert)``
    in expert-major order, so the dispatcher signatures inherited from
    ``LocalTokenDispatcher`` don't fit. The overrides below carry
    ``# pyrefly: ignore [bad-override]`` for the same reason
    ``DeepEPTokenDispatcher`` does.

    ``dispatch`` sorts tokens by global expert, all-to-all dispatches them to
    the EP ranks, and synthesizes a "mock" per-token routing where each
    routed token has top_k=1 (it goes to exactly one local expert) so the
    HF for-loop runs unmodified. ``combine`` reverses the all-to-all,
    unsorts via the saved permutation, and sums across the original top_k.

    ``ep_group`` and the EP sizes are populated by ``HFExpertParallel``'s
    ``_partition_fn`` at parallelize time, mirroring how native
    ``ExpertParallel`` populates ``ep_group`` on its dispatcher.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # Set by HFExpertParallel._partition_fn at parallelize time.
        self.ep_group: dist.ProcessGroup | None = None
        self.ep_size: int | None = None
        self.num_local_experts: int | None = None
        # Per-call state set in dispatch(), consumed in combine().
        # Not thread-safe — assumes serial forward calls (same constraint
        # as DeepEPTokenDispatcher carrying state across dispatch/combine).
        self._input_splits: list[int] | None = None
        self._output_splits: list[int] | None = None
        self._inv_perm: torch.Tensor | None = None
        self._num_tokens: int | None = None
        self._top_k: int | None = None

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sort tokens by expert, all-to-all dispatch, build local routing.

        Args:
            hidden_states: ``(num_tokens, hidden_dim)``
            top_k_index: ``(num_tokens, top_k)`` global expert indices
            top_k_weights: ``(num_tokens, top_k)`` routing weights

        Returns:
            ``(dispatched_tokens, local_top_k_index, local_top_k_weights)``
                - dispatched_tokens: ``(num_received, hidden_dim)``
                - local_top_k_index: ``(num_received, 1)`` local expert indices
                - local_top_k_weights: ``(num_received, 1)`` routing weights
        """
        assert self.ep_group is not None, (
            "ep_group must be set before dispatch. "
            "HFExpertParallel._partition_fn() should set it."
        )
        ep_size = self.ep_size
        num_local = self.num_local_experts
        global_num_experts = self.num_experts
        ep_group = self.ep_group

        num_tokens = hidden_states.size(0)
        top_k = top_k_index.size(-1)

        # Flatten token-expert pairs
        token_idx = (
            torch.arange(num_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )
        expert_ids = top_k_index.reshape(-1)
        sample_weights = top_k_weights.reshape(-1)

        # Sort by expert
        perm = torch.argsort(expert_ids, stable=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=hidden_states.device)

        routed_input = hidden_states[token_idx[perm]]
        sorted_weights = sample_weights[perm]

        # Compute num_tokens_per_expert
        num_tokens_per_expert = torch.histc(
            expert_ids[perm].int(),
            bins=global_num_experts,
            min=0,
            max=global_num_experts - 1,
        )

        # Exchange per-expert token counts
        with torch.no_grad():
            input_splits_t = num_tokens_per_expert.view(ep_size, num_local).sum(dim=1)
            output_splits_t = torch.empty_like(input_splits_t)
            torch.distributed.all_to_all_single(
                output_splits_t, input_splits_t, group=ep_group
            )
            self._input_splits = input_splits_t.int().tolist()
            self._output_splits = output_splits_t.int().tolist()

        # Dispatch tokens
        dispatched = all_to_all_single_autograd(
            routed_input, self._output_splits, self._input_splits, ep_group
        )

        # Dispatch routing weights alongside tokens
        dispatched_weights = all_to_all_single_autograd(
            sorted_weights.unsqueeze(-1),
            self._output_splits,
            self._input_splits,
            ep_group,
        ).squeeze(-1)

        # Exchange per-expert counts for local routing
        with torch.no_grad():
            local_ntpe_tensor = torch.empty_like(num_tokens_per_expert)
            torch.distributed.all_to_all_single(
                local_ntpe_tensor, num_tokens_per_expert, group=ep_group
            )

        # Build local mock routing (matches by-source-rank token ordering)
        local_ntpe_per_source = local_ntpe_tensor.view(ep_size, num_local)
        local_expert_indices = torch.repeat_interleave(
            torch.arange(num_local, device=hidden_states.device).repeat(ep_size),
            local_ntpe_per_source.reshape(-1).long(),
        )
        mock_top_k_index = local_expert_indices.unsqueeze(1)
        mock_top_k_weights = dispatched_weights.unsqueeze(1)

        # Save state for combine
        self._inv_perm = inv_perm
        self._num_tokens = num_tokens
        self._top_k = top_k

        return dispatched, mock_top_k_index, mock_top_k_weights

    # pyrefly: ignore [bad-override]
    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        """All-to-all combine, unsort, sum across top_k.

        Returns:
            ``(num_tokens, hidden_dim)`` accumulated output.
        """
        combined = all_to_all_single_autograd(
            expert_output,
            self._input_splits,  # reversed
            self._output_splits,  # reversed
            self.ep_group,
        )
        combined = combined[self._inv_perm]
        return combined.view(self._num_tokens, self._top_k, -1).sum(dim=1)
