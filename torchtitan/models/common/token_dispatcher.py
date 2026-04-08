# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)

from torchtitan.distributed.expert_parallel import _permute, _unpermute
from torchtitan.ops.scatter_add import deterministic_scatter_add


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """Metadata returned by LocalTokenDispatcher.dispatch() for use in combine()."""

    token_indices_experts_sorted: torch.Tensor  # (N*top_k,)
    top_scores: torch.Tensor  # (N, top_k) original scores
    top_scores_experts_sorted: torch.Tensor  # (N*top_k,) scores in expert-sorted order


@dataclass(frozen=True, kw_only=True)
class DispatchMetadata(LocalDispatchMetadata):
    """Metadata returned by TokenDispatcher.dispatch() for use in combine()."""

    input_shape: tuple  # for _unpermute
    permuted_indices: torch.Tensor  # for _unpermute
    input_splits: list[int]
    output_splits: list[int]


class BaseTokenDispatcher:
    """Abstract base for token dispatchers in MoE.

    A token dispatcher handles the full token routing lifecycle:
    dispatch (reorder + optional EP all-to-all) and combine (reverse).

    Not an nn.Module — dispatchers have no learnable parameters or buffers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        num_experts: int
        top_k: int
        score_before_experts: bool

        def build(self):
            raise NotImplementedError(
                f"{type(self).__name__}.build() is abstract. "
                "Use a concrete Config subclass (e.g. LocalTokenDispatcher.Config)."
            )

    def __init__(self, config: Config):
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.score_before_experts = config.score_before_experts

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder and (optionally) dispatch tokens to experts.

        Args:
            x: (num_tokens, dim) all input tokens
            top_scores: (num_tokens, top_k) routing scores
            selected_experts_indices: (num_tokens, top_k) expert indices per token
            num_tokens_per_expert: (num_experts,) token counts per expert

        Returns:
            routed_input: (R, dim) tokens ready for expert computation, padded for alignment
            num_tokens_per_expert_local: (num_local_experts,) aligned token counts
            metadata: state needed by combine()
        """
        raise NotImplementedError

    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: LocalDispatchMetadata,
    ) -> torch.Tensor:
        """Combine expert outputs back to original token order.

        Args:
            routed_output: (R, dim) expert outputs
            metadata: state from dispatch()

        Returns:
            output: (num_tokens, dim) combined output
        """
        raise NotImplementedError


class LocalTokenDispatcher(BaseTokenDispatcher):
    """Token dispatcher for EP=1. Handles local token reordering only."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenDispatcher.Config):
        def build(self) -> "LocalTokenDispatcher":
            return LocalTokenDispatcher(self)

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
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

        # shape (bs*slen*top_k, dim)
        routed_input = x[token_indices_experts_sorted]

        # Apply scores before expert computation if configured
        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted=token_indices_experts_sorted,
            top_scores=top_scores,
            top_scores_experts_sorted=top_scores_experts_sorted,
        )
        return routed_input, num_tokens_per_expert, metadata

    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: LocalDispatchMetadata,
    ) -> torch.Tensor:
        num_tokens = metadata.token_indices_experts_sorted.shape[0] // self.top_k
        dim = routed_output.shape[1]

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * metadata.top_scores_experts_sorted.reshape(-1, 1)
            ).to(routed_output.dtype)

        out = torch.zeros(
            num_tokens, dim, dtype=routed_output.dtype, device=routed_output.device
        )
        return deterministic_scatter_add(
            out,
            metadata.token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim),
            routed_output,
        )


class TokenDispatcher(BaseTokenDispatcher):
    """Token dispatcher for EP>1. Handles token reorder + all-to-all dispatch/combine.

    ep_group is set by the parallelization code after construction.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenDispatcher.Config):
        def build(self) -> "TokenDispatcher":
            return TokenDispatcher(self)

    def __init__(self, config: Config):
        super().__init__(config)
        # Set by parallelization code (e.g. apply_moe_ep_tp)
        self.ep_group: dist.ProcessGroup

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DispatchMetadata]:
        ep_degree = dist.get_world_size(self.ep_group)

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

        # shape (bs*slen*top_k, dim)
        routed_input = x[token_indices_experts_sorted]

        # Apply scores before expert computation if configured
        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # All-to-all of per-expert token counts
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=self.ep_group,
            )
            # Wait explicitly for use by triton kernel later
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this incurs a device-to-host sync
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            input_splits_list = input_splits.tolist()
            output_splits_list = output_splits.tolist()

        # All-to-all dispatch tokens to EP ranks
        routed_input = all_to_all_single_autograd(
            routed_input,
            output_splits_list,
            input_splits_list,
            self.ep_group,
        )

        # Reorder tokens from rank-major to expert-major layout for grouped_mm
        num_local_experts = num_tokens_per_expert_group.shape[0] // ep_degree
        (
            input_shape,
            routed_input,
            permuted_indices,
            num_tokens_per_expert_group,
        ) = _permute(
            routed_input,
            num_tokens_per_expert_group,
            ep_degree,
            num_local_experts,
        )

        metadata = DispatchMetadata(
            token_indices_experts_sorted=token_indices_experts_sorted,
            top_scores=top_scores,
            top_scores_experts_sorted=top_scores_experts_sorted,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        return routed_input, num_tokens_per_expert_group, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: DispatchMetadata,
    ) -> torch.Tensor:
        num_tokens = metadata.token_indices_experts_sorted.shape[0] // self.top_k
        dim = routed_output.shape[1]

        # Reverse expert-major reordering
        routed_output = _unpermute(
            routed_output, metadata.input_shape, metadata.permuted_indices
        )

        # All-to-all combine: send tokens back to originating EP ranks
        routed_output = all_to_all_single_autograd(
            routed_output,
            metadata.input_splits,
            metadata.output_splits,
            self.ep_group,
        )

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * metadata.top_scores_experts_sorted.reshape(-1, 1)
            ).to(routed_output.dtype)

        out = torch.zeros(
            num_tokens, dim, dtype=routed_output.dtype, device=routed_output.device
        )
        return deterministic_scatter_add(
            out,
            metadata.token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim),
            routed_output,
        )


@dataclass(frozen=True, kw_only=True)
class DeepEPDispatchMetadata:
    """Metadata for DeepEP/HybridEP token dispatch."""

    state: object  # deepep.DispatchState or hybridep.DispatchState


class DeepEPTokenDispatcher(BaseTokenDispatcher):
    """Token dispatcher using DeepEP/HybridEP for efficient token dispatch/combine.

    Uses DeepEP library kernels instead of standard all-to-all collectives for
    token dispatch and combine. For the DeepEP backend, combine is asynchronous
    — callers must call sync_combine() before using the result.

    ep_group is set by the parallelization code after construction.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenDispatcher.Config):
        """Config for DeepEP/HybridEP token dispatcher.

        Args:
            comm_backend: "deepep" for H100/NVLink Switch, "hybridep" for GB200/NVLink72.
            hybridep_non_blocking_expert_capacity_factor: None = blocking mode (default).
                float in (0, 1] = non-blocking mode; controls the fused-permute
                output tensor size (num_permuted_tokens). Only used with hybridep.
            pad_multiple: Alignment size for token groups needed by quantized grouped
                GEMMs (e.g. 16 for FP8, 32 for MXFP8). Only supported with hybridep.
                None means no padding.
        """

        comm_backend: str
        hybridep_non_blocking_expert_capacity_factor: float | None = None
        pad_multiple: int | None = None

        def build(self) -> "DeepEPTokenDispatcher":
            return DeepEPTokenDispatcher(self)

    def __init__(self, config: Config):
        super().__init__(config)
        self.comm_backend = config.comm_backend
        self.hybridep_non_blocking_expert_capacity_factor = (
            config.hybridep_non_blocking_expert_capacity_factor
        )
        self.pad_multiple = config.pad_multiple
        # Set by parallelization code (e.g. apply_moe_ep_tp)
        self.ep_group: dist.ProcessGroup

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        if config.comm_backend == "hybridep":
            from torchtitan.distributed.deepep import hybridep  # noqa: F401
        else:
            from torchtitan.distributed.deepep import deepep  # noqa: F401

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        ep_degree = dist.get_world_size(self.ep_group)
        num_local_experts = self.num_experts // ep_degree

        if self.comm_backend == "hybridep":
            from torchtitan.distributed.deepep.hybridep import dispatch_tokens

            hidden, tokens_per_expert, state = dispatch_tokens(
                x,
                selected_experts_indices,
                top_scores,
                num_local_experts,
                self.num_experts,
                self.ep_group,
                score_before_experts=self.score_before_experts,
                non_blocking_expert_capacity_factor=self.hybridep_non_blocking_expert_capacity_factor,
                pad_multiple=self.pad_multiple,
            )
        else:
            from torchtitan.distributed.deepep.deepep import dispatch_tokens

            hidden, tokens_per_expert, state = dispatch_tokens(
                x,
                selected_experts_indices,
                top_scores,
                num_local_experts,
                self.num_experts,
                self.ep_group,
                score_before_experts=self.score_before_experts,
            )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden, tokens_per_expert, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
    ) -> torch.Tensor:
        if self.comm_backend == "hybridep":
            from torchtitan.distributed.deepep import hybridep

            return hybridep.combine_tokens(
                routed_output,
                metadata.state,  # pyrefly: ignore [bad-argument-type]
                pad_multiple=self.pad_multiple,
            )
        else:
            from torchtitan.distributed.deepep.deepep import combine_tokens

            # pyrefly: ignore [bad-argument-type]
            return combine_tokens(routed_output, metadata.state)
