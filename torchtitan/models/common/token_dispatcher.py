# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import DeviceMesh

from torchtitan.config import Configurable
from torchtitan.ops.scatter_add import deterministic_scatter_add


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """Metadata returned by LocalTokenDispatcher.dispatch() for use in combine()."""

    token_indices_experts_sorted: torch.Tensor  # (N*top_k,)
    top_scores_experts_sorted: torch.Tensor  # (N*top_k,) scores in expert-sorted order


@dataclass(frozen=True, kw_only=True)
class AllToAllDispatchMetadata(LocalDispatchMetadata):
    """Metadata returned by AllToAllTokenDispatcher.dispatch() for use in combine()."""

    input_shape: tuple  # for _unpermute
    permuted_indices: torch.Tensor  # for _unpermute
    input_splits: list[int]
    output_splits: list[int]


class LocalTokenDispatcher(Configurable):
    """Token dispatcher for EP=1. Handles local token reordering only.

    Also serves as the base class for EP dispatchers (AllToAllTokenDispatcher,
    DeepEPTokenDispatcher) which override dispatch() and combine().

    Not an nn.Module — dispatchers have no learnable parameters or buffers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        num_experts: int
        top_k: int
        score_before_experts: bool = True

    def __init__(self, config: Config):
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.score_before_experts = config.score_before_experts

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert assignment for local expert computation.

        Args:
            x: (num_tokens, dim) all input tokens
            top_scores: (num_tokens, top_k) routing scores
            selected_experts_indices: (num_tokens, top_k) expert indices per token

        Returns:
            routed_input: (num_tokens * top_k, dim) tokens sorted by expert index
            num_tokens_per_expert: (num_experts,) token counts per expert
            metadata: LocalDispatchMetadata for combine()
        """
        # TODO: Extract this local reordering block (histc, argsort, score
        # application) into a shared helper — it's duplicated in
        # AllToAllTokenDispatcher.dispatch.
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
        x: torch.Tensor,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        """Score, scatter_add, and optionally overlap shared_experts.

        For EP=1, no communication is needed. shared_experts runs before
        scatter_add (no async overlap benefit here, but keeps the interface
        uniform with AllToAllTokenDispatcher).

        Args:
            routed_output: (num_tokens * top_k, dim) expert outputs
            metadata: LocalDispatchMetadata from dispatch()
            x: (num_tokens, dim) original input tokens
            shared_experts: optional shared expert module to overlap

        Returns:
            (num_tokens, dim) combined output with shared_experts added.
        """
        out = shared_experts(x) if shared_experts is not None else torch.zeros_like(x)

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * metadata.top_scores_experts_sorted.reshape(-1, 1)
            ).to(routed_output.dtype)

        dim = x.shape[-1]
        out = deterministic_scatter_add(
            out,
            metadata.token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim),
            routed_output,
        )
        return out


class AllToAllTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher for EP>1. Handles token reorder + all-to-all dispatch/combine.

    Handles the full token routing lifecycle:
    dispatch (reorder + EP all-to-all) and combine (reverse).

    ep_mesh is set by the parallelization code after construction.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # DeviceMesh (not ProcessGroup) so that CooR precompile can use
        # torch.ops._dtensor.mesh_get_process_group to keep the FX graph
        # rank-agnostic. Set by ExpertParallel._partition_fn().
        self.ep_mesh: DeviceMesh | None = None
        # TODO: these should be set at config time
        # Set at runtime by apply_moe_ep_tp. Uses _sym_get_coordinate
        # so the rank is a SymInt under CooR precompile.
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt = -1

    def _split_along_sp(self, *tensors: torch.Tensor) -> list[torch.Tensor]:
        """Split tensors along the first dim across EP ranks for sequence parallel."""
        sp_size = self.sp_size
        sp_rank = self.sp_rank
        results = []
        for t in tensors:
            assert t.is_contiguous()
            num_tokens = t.shape[0]
            if num_tokens % sp_size != 0:
                raise ValueError(
                    "Uneven split of tokens is not supported yet. "
                    "Requires EP degree dividing batch size * seq len."
                )
            local_num_tokens = num_tokens // sp_size
            offset = sp_rank * local_num_tokens
            results.append(t[offset : offset + local_num_tokens])
        return results

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AllToAllDispatchMetadata]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        When sp_size > 1 (sequence parallel), inputs are first split along
        the token dim so each EP rank processes a disjoint subset.

        Args:
            x: (num_tokens, dim) all input tokens (global if sp_size > 1)
            top_scores: (num_tokens, top_k) routing scores
            selected_experts_indices: (num_tokens, top_k) expert indices per token

        Returns:
            routed_input: (R, dim) tokens in expert-major order for local experts
            num_tokens_per_expert_local: (num_local_experts,) token counts
            metadata: AllToAllDispatchMetadata for combine()
        """
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )
        ep_size = self.ep_mesh.size()

        if self.sp_size > 1:
            assert self.sp_rank >= 0, (
                "sp_rank must be set before use. "
                "apply_moe_ep_tp() should set it from tp_mesh._sym_get_coordinate()."
            )
            # NOTE: If needed, we can pad tokens in case bs*slen is not divisible by TP degree
            # shape (batch_size * seq_len // ep_size, top_k)
            x, top_scores, selected_experts_indices = self._split_along_sp(
                x, top_scores, selected_experts_indices
            )

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
                group=self.ep_mesh,
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            # non_blocking=True is safe in eager, but under torch.compile the
            # async D2H transfer can race with the subsequent .tolist()/.item()
            # calls, producing stale values and failing unbacked-symint guards.
            non_blocking = not torch.compiler.is_compiling()
            input_splits = (
                num_tokens_per_expert.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=non_blocking)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_tokens_per_expert_group.view(ep_size, -1)
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
            self.ep_mesh,
        )

        # Reorder from rank-major to expert-major via _permute.
        #
        # num_tokens_per_expert_group layout after all-to-all:
        #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        # _permute reshuffles to:
        #   (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
        num_local_experts = num_tokens_per_expert_group.shape[0] // ep_size
        (
            input_shape,
            routed_input,
            permuted_indices,
            num_tokens_per_expert_group,
        ) = self._permute(
            routed_input,
            num_tokens_per_expert_group,
            ep_size,
            num_local_experts,
        )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted=token_indices_experts_sorted,
            top_scores_experts_sorted=top_scores_experts_sorted,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        return routed_input, num_tokens_per_expert_group, metadata

    def _permute(
        self, routed_input, num_tokens_per_expert_group, ep_size, num_local_experts
    ):
        """Reorder tokens from rank-major to expert-major layout.

        Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
        """
        device = num_tokens_per_expert_group.device
        total = num_tokens_per_expert_group.sum()

        # [R, E] matrix of token counts per (rank, expert)
        t_mat = num_tokens_per_expert_group.view(ep_size, num_local_experts)

        # Where each (r, e) segment starts in the input (rank-major order)
        input_starts = (
            num_tokens_per_expert_group.cumsum(0) - num_tokens_per_expert_group
        ).view(ep_size, num_local_experts)

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
            + torch.arange(total, device=device)
            - output_starts[seg_ids]
        )

        num_tokens_per_expert = t_mat.sum(0)
        return (
            routed_input.shape,
            routed_input[permuted_indices, :],
            permuted_indices,
            num_tokens_per_expert,
        )

    def _unpermute(self, routed_output, input_shape, permuted_indices):
        """Reverse expert-major reordering."""
        out_unpermuted = routed_output.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = routed_output
        return out_unpermuted

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x: torch.Tensor,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + score + scatter_add.

        shared_experts overlaps with the async all-to-all combine — it runs
        while the a2a is in flight on the NCCL stream. scatter_add forces sync.

        When sp_size > 1 (sequence parallel), dispatch uses local token
        indices. Combine offsets them to global positions so scatter_add
        into full x is correct.

        Args:
            routed_output: (R, dim) expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x: (num_tokens, dim) original input tokens
            shared_experts: optional shared expert module to overlap

        Returns:
            (num_tokens, dim) combined output with shared_experts added.
        """
        # Reverse expert-major reordering
        routed_output = self._unpermute(
            routed_output, metadata.input_shape, metadata.permuted_indices
        )

        assert self.ep_mesh is not None, (
            "ep_mesh must be set before combine. "
            "ExpertParallel._partition_fn() should set it."
        )
        # All-to-all combine: returns AsyncCollectiveTensor — the a2a runs
        # on the NCCL stream and won't block until the tensor is accessed.
        routed_output = all_to_all_single_autograd(
            routed_output,
            metadata.input_splits,
            metadata.output_splits,
            self.ep_mesh,
        )

        # shared_experts overlaps with the async a2a (NCCL stream).
        # Score application + scatter_add forces the a2a to sync.
        out = shared_experts(x) if shared_experts is not None else torch.zeros_like(x)

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * metadata.top_scores_experts_sorted.reshape(-1, 1)
            ).to(routed_output.dtype)

        # When sequence_parallel is active, dispatch splits tokens to a local
        # shard, so token_indices_experts_sorted are 0-based local indices.
        # Offset them to global positions so scatter_add into full x is correct.
        if self.sp_size > 1:
            local_num_tokens = x.shape[0] // self.sp_size
            token_indices_experts_sorted = (
                metadata.token_indices_experts_sorted + local_num_tokens * self.sp_rank
            )
        else:
            token_indices_experts_sorted = metadata.token_indices_experts_sorted

        out = deterministic_scatter_add(
            out,
            token_indices_experts_sorted.reshape(-1, 1).expand(-1, x.shape[-1]),
            routed_output,
        )
        return out


class TorchAOTokenDispatcher(AllToAllTokenDispatcher):
    """Token dispatcher for EP>1 with token group padding for quantized grouped GEMMs.

    Uses torchao's ``permute_and_pad`` instead of the standard ``_permute`` to
    reorder tokens into expert-major order and pad each expert's token group to
    a multiple of ``pad_multiple``. This alignment is required by FP8/MXFP8
    quantized grouped GEMM kernels (e.g. 16 for FP8, 32 for MXFP8).

    ep_mesh is set by ExpertParallel / ExpertTensorParallel._apply().
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pad_multiple: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.pad_multiple = config.pad_multiple

    def _permute(
        self, routed_input, num_tokens_per_expert_group, ep_size, num_local_experts
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
            ep_size,
            num_local_experts,
            self.pad_multiple,
        )
        return (
            input_shape,
            routed_input,
            permuted_indices,
            num_tokens_per_expert_group_padded,
        )

    def _unpermute(self, routed_output, input_shape, permuted_indices):
        # Strip the padding sentinel row added by permute_and_pad
        out_unpermuted = routed_output.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = routed_output
        return out_unpermuted[:-1]


@dataclass(frozen=True, kw_only=True)
class DeepEPDispatchMetadata:
    """Metadata for DeepEP/HybridEP token dispatch."""

    state: object  # deepep.DispatchState or hybridep.DispatchState


class DeepEPTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher using DeepEP/HybridEP for efficient token dispatch/combine.

    Uses DeepEP library kernels instead of standard all-to-all collectives for
    token dispatch and combine. For the DeepEP backend, combine is asynchronous
    — callers must call sync_combine() before using the result.

    ep_mesh is set by ExpertParallel / ExpertTensorParallel._apply().
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        """Config for DeepEP/HybridEP token dispatcher.

        Args:
            comm_backend: "deepep" for H100/NVLink Switch, "hybridep" for GB200/NVLink72.
            non_blocking_capacity_factor: Enable non-blocking HybridEP dispatch with a
                given capacity factor.

                Setting this to a float in (0, 1] enables CPU-free non-blocking
                dispatch and controls num_permuted_tokens — the fused-permute
                output capacity, estimated as:
                num_tokens × ep_size × min(num_local_experts, top_k) × cf,
                aligned for MXFP8.  Tokens whose permuted offset exceeds this
                limit are silently dropped (overflow_flag is set on GPU).

                - None = blocking mode (default).  HybridEP calls
                  cudaStreamSynchronize after dispatch, copies
                  tokens_per_expert to pinned CPU memory, and computes the
                  exact num_permuted_tokens on the host.  No token dropping.
                - 1.0 = non-blocking, worst-case sizing: every token can reach
                  every local expert, no drops, highest memory.
                - < 1.0 = non-blocking, reduced memory; controls the
                  fused-permute output tensor size (num_permuted_tokens).
                  Safe in practice when forced load balancing (e.g. aux-loss /
                  round-robin) keeps distribution roughly uniform.

                Note: this factor has no lasting effect on the all-to-all
                communication buffer.  HybridEP's dispatch_with_permute
                internally passes the actual num_tokens to
                update_template_config, which auto-grows the buffer to the
                full token count on the first dispatch regardless of this
                setting.
        """

        comm_backend: str
        non_blocking_capacity_factor: float | None = None
        pad_multiple: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.comm_backend = config.comm_backend
        self.non_blocking_capacity_factor = config.non_blocking_capacity_factor
        self.pad_multiple = config.pad_multiple
        # Set by ExpertParallel / ExpertTensorParallel._partition_fn()
        self.ep_mesh: DeviceMesh | None = None

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
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()
        if self.comm_backend == "hybridep":
            from torchtitan.distributed.deepep.hybridep import dispatch_tokens

            hidden_states, tokens_per_expert, state = dispatch_tokens(
                x,
                selected_experts_indices,
                top_scores,
                num_local_experts,
                self.num_experts,
                ep_group,
                score_before_experts=self.score_before_experts,
                non_blocking_expert_capacity_factor=self.non_blocking_capacity_factor,
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
                ep_group,
                score_before_experts=self.score_before_experts,
            )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states, tokens_per_expert, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x: torch.Tensor,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        """Combine tokens via DeepEP/HybridEP, overlapping shared_experts.

        For the deepep backend, combine is async — shared_experts runs while
        the combine is in flight, then sync_combine() waits before the addition.
        """
        if self.comm_backend == "hybridep":
            from torchtitan.distributed.deepep import hybridep

            routed_output = hybridep.combine_tokens(
                routed_output,
                metadata.state,  # pyrefly: ignore [bad-argument-type]
                pad_multiple=self.pad_multiple,
            )
        else:
            from torchtitan.distributed.deepep.deepep import combine_tokens

            # pyrefly: ignore [bad-argument-type]
            routed_output = combine_tokens(routed_output, metadata.state)

        # shared_experts runs in parallel with combine communication.
        # This is the key optimization - we overlap compute with communication.
        shared_out = shared_experts(x) if shared_experts is not None else None

        # Sync the combine operation before using routed_output.
        # This inserts a CUDA stream wait, ensuring combine is complete before
        # the subsequent addition or reshape operations read routed_output.
        from torchtitan.distributed.deepep.deepep import sync_combine

        sync_combine()

        if shared_out is not None:
            routed_output = routed_output + shared_out
        return routed_output
