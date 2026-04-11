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

from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """Metadata returned by LocalTokenDispatcher.dispatch() for use in combine()."""

    token_indices_experts_sorted: torch.Tensor  # (N*top_k,)
    top_scores_experts_sorted: torch.Tensor  # (N*top_k,) scores in expert-sorted order


@dataclass(frozen=True, kw_only=True)
class DispatchMetadata(LocalDispatchMetadata):
    """Metadata returned by TokenDispatcher.dispatch() for use in combine()."""

    input_shape: tuple  # for _unpermute
    permuted_indices: torch.Tensor  # for _unpermute
    input_splits: list[int]
    output_splits: list[int]


class BaseTokenDispatcher(Configurable):
    """Abstract base for token dispatchers in MoE.

    A token dispatcher handles the full token routing lifecycle:
    dispatch (reorder + optional EP all-to-all) and combine (reverse).

    Not an nn.Module — dispatchers have no learnable parameters or buffers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        num_experts: int
        top_k: int
        score_before_experts: bool
        ep_degree: int = 1

    def __init__(self, config: Config):
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.score_before_experts = config.score_before_experts
        self.ep_degree = config.ep_degree

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder and (optionally) dispatch tokens to experts.

        Args:
            x: (num_tokens, dim) all input tokens
            top_scores: (num_tokens, top_k) routing scores
            selected_experts_indices: (num_tokens, top_k) expert indices per token

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
        """Reverse the dispatch: unpermute + optional all-to-all.

        Args:
            routed_output: (R, dim) expert outputs
            metadata: state from dispatch()

        Returns:
            routed_output in local token order (not yet scatter_add'd).
            For EP, this is an AsyncCollectiveTensor — the a2a runs on
            the NCCL stream and won't block until accessed.
        """
        raise NotImplementedError


class LocalTokenDispatcher(BaseTokenDispatcher):
    """Token dispatcher for EP=1. Handles local token reordering only."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenDispatcher.Config):
        pass

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        # TODO: Extract this local reordering block (histc, argsort, score
        # application) into a shared helper — it's duplicated in
        # TokenDispatcher.dispatch.
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
            top_scores_experts_sorted=top_scores_experts_sorted,
        )
        return routed_input, num_tokens_per_expert, metadata

    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: LocalDispatchMetadata,
    ) -> torch.Tensor:
        return routed_output


class TokenDispatcher(BaseTokenDispatcher):
    """Token dispatcher for EP>1. Handles token reorder + all-to-all dispatch/combine.

    ep_group is set by the parallelization code after construction.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # Set by ExpertParallel / ExpertTensorParallel._apply()
        self.ep_group: dist.ProcessGroup

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DispatchMetadata]:
        ep_degree = self.ep_degree

        # TODO: Extract this local reordering block (histc, argsort, score
        # application) into a shared helper — it's duplicated in
        # LocalTokenDispatcher.dispatch.
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

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=self.ep_group,
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
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

        # Reorder from rank-major to expert-major via _permute.
        #
        # num_tokens_per_expert_group layout after all-to-all:
        #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        # _permute reshuffles to:
        #   (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
        num_local_experts = num_tokens_per_expert_group.shape[0] // ep_degree
        (
            input_shape,
            routed_input,
            permuted_indices,
            num_tokens_per_expert_group,
        ) = self._permute(
            routed_input,
            num_tokens_per_expert_group,
            ep_degree,
            num_local_experts,
        )

        metadata = DispatchMetadata(
            token_indices_experts_sorted=token_indices_experts_sorted,
            top_scores_experts_sorted=top_scores_experts_sorted,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        return routed_input, num_tokens_per_expert_group, metadata

    def _permute(
        self, routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
    ):
        """Reorder tokens from rank-major to expert-major layout.

        Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
        """
        device = num_tokens_per_expert_group.device
        total = num_tokens_per_expert_group.sum()

        # [R, E] matrix of token counts per (rank, expert)
        t_mat = num_tokens_per_expert_group.view(ep_degree, num_local_experts)

        # Where each (r, e) segment starts in the input (rank-major order)
        input_starts = (
            num_tokens_per_expert_group.cumsum(0) - num_tokens_per_expert_group
        ).view(ep_degree, num_local_experts)

        # Transpose to expert-major [E, R] and flatten
        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        # For each output position, find its input position:
        #   output[p] = input[input_starts[seg] + (p - output_starts[seg])]
        seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(
            segment_lens
        )
        output_starts = segment_lens.cumsum(0) - segment_lens
        permuted_indices = (
            input_starts[seg_ids]
            + torch.arange(
                total, device=device
            )  # pyrefly: ignore [no-matching-overload]
            - output_starts[seg_ids]
        )

        num_tokens_per_expert = t_mat.sum(0)
        return (
            routed_input.shape,
            routed_input[permuted_indices, :],
            permuted_indices,
            num_tokens_per_expert,
        )

    def _unpermute(self, out, input_shape, permuted_indices):
        """Reverse expert-major reordering."""
        out_unpermuted = out.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = out
        return out_unpermuted

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: DispatchMetadata,
    ) -> torch.Tensor:
        # Reverse expert-major reordering
        routed_output = self._unpermute(
            routed_output, metadata.input_shape, metadata.permuted_indices
        )

        # All-to-all combine: send tokens back to originating EP ranks.
        # Returns an AsyncCollectiveTensor — the a2a runs on the NCCL stream
        # and won't block until the tensor is accessed (e.g. by scatter_add
        # in MoE.forward).
        return all_to_all_single_autograd(
            routed_output,
            metadata.input_splits,
            metadata.output_splits,
            self.ep_group,
        )


class TorchAOTokenDispatcher(TokenDispatcher):
    """Token dispatcher for EP>1 with token group padding for quantized grouped GEMMs.

    Uses torchao's ``permute_and_pad`` instead of the standard ``_permute`` to
    reorder tokens into expert-major order and pad each expert's token group to
    a multiple of ``pad_multiple``. This alignment is required by FP8/MXFP8
    quantized grouped GEMM kernels (e.g. 16 for FP8, 32 for MXFP8).

    ep_group is set by ExpertParallel / ExpertTensorParallel._apply().
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TokenDispatcher.Config):
        pad_multiple: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.pad_multiple = config.pad_multiple

    def _permute(
        self, routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
    ):
        # FP8/MXFP8 require groups to be permuted to expert major order AND
        # padded to nearest multiple of 16.
        # It also does padding to make sure the number of tokens each expert
        # gets locally is a multiple of `self.pad_multiple`.
        # Note that this will create side effects when wrapping the for-loop
        # implementation of GroupedExperts, as it does not need padding.
        from torchao.prototype.moe_training.ep.permute import permute_and_pad

        (
            input_shape,
            routed_input,
            permuted_indices,
            num_tokens_per_expert_group_padded,
            _group_offsets,
        ) = permute_and_pad(
            routed_input,
            num_tokens_per_expert_group,
            ep_degree,
            num_local_experts,
            self.pad_multiple,
        )
        return (
            input_shape,
            routed_input,
            permuted_indices,
            num_tokens_per_expert_group_padded,
        )

    def _unpermute(self, routed_output, metadata):
        # Strip the padding sentinel row added by permute_and_pad
        out_unpermuted = routed_output.new_empty(metadata.input_shape)
        out_unpermuted[metadata.permuted_indices, :] = routed_output
        return out_unpermuted[:-1]


@dataclass(frozen=True, kw_only=True)
class DeepEPDispatchMetadata:
    """Metadata for DeepEP/HybridEP token dispatch."""

    state: object  # deepep.DispatchState or hybridep.DispatchState


class DeepEPTokenDispatcher(BaseTokenDispatcher):
    """Token dispatcher using DeepEP/HybridEP for efficient token dispatch/combine.

    Uses DeepEP library kernels instead of standard all-to-all collectives for
    token dispatch and combine. For the DeepEP backend, combine is asynchronous
    — callers must call sync_combine() before using the result.

    ep_group is set by ExpertParallel / ExpertTensorParallel._apply().
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

    def __init__(self, config: Config):
        super().__init__(config)
        self.comm_backend = config.comm_backend
        self.hybridep_non_blocking_expert_capacity_factor = (
            config.hybridep_non_blocking_expert_capacity_factor
        )
        self.pad_multiple = config.pad_multiple
        # Set by ExpertParallel / ExpertTensorParallel._apply()
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
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        num_local_experts = self.num_experts // self.ep_degree

        if self.comm_backend == "hybridep":
            from torchtitan.distributed.deepep.hybridep import dispatch_tokens

            hidden_states, tokens_per_expert, state = dispatch_tokens(
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

            hidden_states, tokens_per_expert, state = dispatch_tokens(
                x,
                selected_experts_indices,
                top_scores,
                num_local_experts,
                self.num_experts,
                self.ep_group,
                score_before_experts=self.score_before_experts,
            )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states, tokens_per_expert, metadata

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
