# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.tensor import DeviceMesh

from torchtitan.config import Configurable
from torchtitan.ops.scatter_add import deterministic_scatter_add


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """Metadata returned by LocalTokenDispatcher.dispatch() for use in combine()."""

    token_indices_experts_sorted_N: torch.Tensor  # noqa: N815
    topk_scores_experts_sorted_N: torch.Tensor  # noqa: N815


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
    DeepEPTokenDispatcher, HybridEPTokenDispatcher) which override
    dispatch() and combine().

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

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """No-op for the EP=1 dispatcher. Subclasses override."""
        del ep_mesh, tp_mesh

    def _local_reorder(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder tokens by expert assignment for local expert computation.

        Groups tokens by expert index via argsort and optionally applies
        routing scores (when ``score_before_experts`` is True).

        Args:
            x_TD: ``(T, D)`` input tokens
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices

        Returns:
            routed_input_ND: ``(N, D)`` where N = T*K. Tokens in expert-sorted
                order, score-weighted if ``score_before_experts``.
            token_indices_experts_sorted_N: ``(N,)`` token-to-original mapping
            topk_scores_experts_sorted_N: ``(N,)`` scores in expert-sorted order
        """
        # Reorder the token indices to match the order of the experts where N = T*K
        token_indices_experts_sorted_N = torch.argsort(
            topk_expert_ids_TK.view(-1), stable=True
        )
        topk_scores_experts_sorted_N = topk_scores_TK.view(-1)[
            token_indices_experts_sorted_N
        ]
        token_indices_experts_sorted_N = token_indices_experts_sorted_N // self.top_k
        routed_input_ND = x_TD[token_indices_experts_sorted_N]

        # Apply scores before expert computation if configured
        if self.score_before_experts:
            routed_input_ND = (
                routed_input_ND.to(torch.float32)
                * topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(x_TD.dtype)

        return (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert assignment for local expert computation.

        Args:
            x_TD: ``(T, D)`` all input tokens
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices per token
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert

        Returns:
            routed_input_RD: ``[R = sum(num_local_tokens_per_expert_E), input_dim(D)]``.
                Tokens sorted by expert index.
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert
            metadata: LocalDispatchMetadata for combine()
        """
        # R = N (no EP all-to-all)
        (
            routed_input_RD,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        )
        return routed_input_RD, num_local_tokens_per_expert_E, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
    ) -> torch.Tensor:
        """Score and scatter_add routed expert outputs.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs
            metadata: LocalDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens
            num_local_tokens_after_padding: Unused for local dispatch; kept
                for a shared dispatcher combine signature.

        Returns:
            out_TD: ``(T, D)`` combined output.
        """
        del num_local_tokens_after_padding
        out_TD = torch.zeros_like(x_TD)

        if not self.score_before_experts:
            routed_output_RD = (
                routed_output_RD.to(torch.float32)
                * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(routed_output_RD.dtype)

        dim = x_TD.shape[-1]
        out_TD = deterministic_scatter_add(
            out_TD,
            metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, dim),
            routed_output_RD,
        )
        return out_TD


class AllToAllTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher for EP>1. Handles token reorder + all-to-all dispatch/combine.

    Handles the full token routing lifecycle:
    dispatch (reorder + EP all-to-all) and combine (reverse).

    ``ep_mesh`` and the ``sp_size`` / ``sp_rank`` SP coordinates are wired
    by the owning ``GroupedExperts.parallelize`` override via
    ``wire_meshes``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # DeviceMesh (not ProcessGroup) so that CooR precompile can use
        # torch.ops._dtensor.mesh_get_process_group to keep the FX graph
        # rank-agnostic. None when EP=1 so dispatch falls back to the
        # LocalTokenDispatcher path.
        self.ep_mesh: DeviceMesh | None = None
        # Sequence-parallel split coordinates derived from tp_mesh.
        # ``sp_rank`` uses ``DeviceMesh._sym_get_coordinate`` so it is a
        # ``SymInt`` under CooR precompile, keeping the FX graph
        # rank-agnostic. Defaults are the TP=1 values.
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt = 0

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh and SP coordinates used by dispatch / combine.

        Both arguments may be ``None`` when the corresponding parallelism
        dimension is disabled; ``dispatch`` / ``combine`` handle the
        disabled cases internally.
        """
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, AllToAllDispatchMetadata | LocalDispatchMetadata
    ]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        When ep_mesh is None (EP=1), falls back to local dispatch — no
        all-to-all communication, just local token reordering with padding.

        With SP, x_TD/topk_scores_TK/topk_expert_ids_TK are already
        the local SP shard (from DTensor Shard to_local via LocalMapConfig).

        Args:
            x_TD: ``(T, D)`` local token shard
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices
            num_local_tokens_per_expert_E: ``(E,)`` token counts for this local
                token shard

        Returns:
            routed_input_RD: ``[R = sum(num_tokens_per_local_expert_e), input_dim(D)]``.
                Tokens in expert-major order for local experts.
            num_tokens_per_local_expert_e: ``(num_local_experts,)`` token counts
            metadata: dispatch metadata for combine()
        """
        # EP=1: fall back to local dispatch (no all-to-all needed)
        if self.ep_mesh is None:
            return super().dispatch(
                x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
            )

        ep_size = self.ep_mesh.size()
        # _local_reorder returns (N, D) where N = T*K.
        # EP all-to-all below produces (R, D) where R != N.
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_global_tokens_per_local_expert_E = all_to_all_single(
                num_local_tokens_per_expert_E,
                None,
                None,
                group=self.ep_mesh,
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_global_tokens_per_local_expert_E = (
                torch.ops._c10d_functional.wait_tensor(
                    num_global_tokens_per_local_expert_E
                )
            )
            input_splits = (
                num_local_tokens_per_expert_E.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_global_tokens_per_local_expert_E.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            input_splits_list = input_splits.tolist()
            output_splits_list = output_splits.tolist()

        # All-to-all dispatch tokens to EP ranks.
        routed_input = all_to_all_single(
            routed_input,
            output_splits_list,
            input_splits_list,
            self.ep_mesh,
        )

        # Reorder from rank-major to expert-major via _permute.
        #
        # num_global_tokens_per_local_expert_E layout after all-to-all
        # (e = local experts, EP = EP ranks):
        #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        # _permute reshuffles to:
        #   (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
        # TODO: Consider using num_global_tokens_per_local_expert_e as the
        # expert_bias_e update buffer, then all-gather on EP ranks. This
        # is blocked by clarification on HybridEP token dropping.
        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        ) = self._permute(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
        )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    def _permute(
        self,
        routed_input_RD,
        num_global_tokens_per_local_expert_E,
    ):
        """Reorder tokens from rank-major to expert-major layout.

        Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)

        Collapses token count matrix ``t_mat`` from ``(EP, e)`` to
        ``num_global_tokens_per_local_expert_e`` ``(e,)`` by summing across ranks.
        """
        # pyrefly: ignore [missing-attribute]
        ep_size = self.ep_mesh.size()
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size
        device = num_global_tokens_per_local_expert_E.device

        # (EP, e) matrix of token counts per (rank, local_expert)
        t_mat = num_global_tokens_per_local_expert_E.view(ep_size, e)

        # Where each (r, e) segment starts in the input (rank-major order)
        input_starts = (
            num_global_tokens_per_local_expert_E.cumsum(0)
            - num_global_tokens_per_local_expert_E
        ).view(ep_size, e)

        # Transpose to expert-major (e, EP) and flatten
        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        # For each output position, find its input position:
        #   output[p] = input[input_starts[seg] + (p - output_starts[seg])]
        seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(
            segment_lens
        )
        output_starts = segment_lens.cumsum(0) - segment_lens
        # seg_ids.shape[0] == segment_lens.sum(), but reuses the unbacked symint
        # already created by repeat_interleave above.
        permuted_indices = (
            input_starts[seg_ids]
            + torch.arange(seg_ids.shape[0], device=device)
            - output_starts[seg_ids]
        )

        num_global_tokens_per_local_expert_e = t_mat.sum(0)
        return (
            routed_input_RD.shape,
            routed_input_RD[permuted_indices, :],
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        )

    def _unpermute(self, routed_output_RD, input_shape, permuted_indices):
        """Reverse expert-major reordering."""
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + score + scatter_add.

        When sp_size > 1, dispatch uses local token indices.
        Combine offsets them to global positions so scatter_add
        into full x_TD is correct.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens
            num_local_tokens_after_padding: Local token count to use for the
                combined SP view after logical padding. MoE padding passes this
                count without materializing pad rows.

        Returns:
            out_TD: Combined output. With SP, shape is
                ``(num_local_tokens_after_padding * sp_size, D)``.
        """
        # EP=1: fall back to local combine (no all-to-all needed)
        if self.ep_mesh is None:
            return super().combine(
                routed_output_RD,
                metadata,
                x_TD,
                num_local_tokens_after_padding=num_local_tokens_after_padding,
            )

        # Reverse expert-major reordering
        routed_output_RD = self._unpermute(
            routed_output_RD, metadata.input_shape, metadata.permuted_indices
        )

        # All-to-all combine: returns AsyncCollectiveTensor — the a2a runs
        # on the NCCL stream and won't block until the tensor is accessed.
        routed_output = all_to_all_single(
            routed_output,
            metadata.input_splits,
            metadata.output_splits,
            self.ep_mesh,
        )

        # With SP, create a full-size buffer for scatter_add so routed results
        # from all SP ranks can be placed at global positions. Padded tail rows
        # are never routed and are sliced off below.
        out_TD = torch.zeros(
            num_local_tokens_after_padding * self.sp_size,
            x_TD.shape[-1],
            device=x_TD.device,
            dtype=x_TD.dtype,
        )

        if not self.score_before_experts:
            routed_output_RD = (
                routed_output_RD.to(torch.float32)
                * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(routed_output_RD.dtype)

        # With SP, token indices are 0-based within the local shard.
        # Offset to global positions for the full-size scatter buffer.
        if self.sp_size > 1:
            token_indices_experts_sorted_N = (
                metadata.token_indices_experts_sorted_N
                + num_local_tokens_after_padding * self.sp_rank
            )
        else:
            token_indices_experts_sorted_N = metadata.token_indices_experts_sorted_N

        assert isinstance(token_indices_experts_sorted_N, torch.Tensor)
        out_TD = deterministic_scatter_add(
            out_TD,
            token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, out_TD.shape[-1]),
            routed_output_RD,
        )
        return out_TD


class TorchAOTokenDispatcher(AllToAllTokenDispatcher):
    """Token dispatcher with token group padding for quantized grouped GEMMs.

    Uses torchao's ``permute_and_pad`` instead of the standard ``_permute`` to
    reorder tokens into expert-major order and pad each expert's token group to
    a multiple of ``pad_multiple``. This alignment is required by FP8/MXFP8
    quantized grouped GEMM kernels (e.g. 16 for FP8, 32 for MXFP8).

    Requires EP to be enabled (ep_mesh must be set). Raises ValueError
    if ep_mesh is None, since quantized grouped GEMMs need padded token
    groups which are only produced by the EP permute_and_pad path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pad_multiple: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.pad_multiple = config.pad_multiple

    def dispatch(
        self, x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
    ):
        if self.ep_mesh is None:
            raise ValueError(
                "TorchAOTokenDispatcher requires expert parallelism (ep_mesh must be set). "
                "Quantized grouped GEMMs need padded token groups, which requires EP>1. "
            )
        return super().dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )

    def _permute(
        self,
        routed_input_RD,
        num_global_tokens_per_local_expert_E,
    ):
        # FP8/MXFP8 require groups to be permuted to expert major order AND
        # padded to nearest multiple of 16.
        # It also does padding to make sure the number of tokens each expert
        # gets locally is a multiple of `self.pad_multiple`.
        # Note that this will create side effects when wrapping the for-loop
        # implementation of GroupedExperts, as it does not need padding.
        from torchao.prototype.moe_training.ep.permute import permute_and_pad

        # pyrefly: ignore [missing-attribute]
        ep_size = self.ep_mesh.size()
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size

        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_padded_e,
            _group_offsets,
        ) = permute_and_pad(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
            ep_size,
            e,
            self.pad_multiple,
        )
        return (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_padded_e,
        )

    def _unpermute(self, routed_output_RD, input_shape, permuted_indices):
        # Strip the padding sentinel row added by permute_and_pad
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD[:-1]


@dataclass(frozen=True, kw_only=True)
class DeepEPDispatchMetadata:
    """Metadata for DeepEP and HybridEP token dispatch."""

    state: object  # deepep.DispatchState or hybridep.DispatchState


class DeepEPTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher using DeepEP for efficient token dispatch/combine.

    Uses DeepEP library kernels (H100/NVLink Switch) instead of standard
    all-to-all collectives. Combine is asynchronous — callers must call
    sync_combine() before using the result.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.ep_mesh: DeviceMesh | None = None

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import deepep  # noqa: F401

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh used by DeepEP dispatch / combine.

        ``tp_mesh`` provides SP coordinates so combine can expand its output
        to full sequence length (matching AllToAll's convention).
        """
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            # pyrefly: ignore [bad-assignment]
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        # Ignore input num_local_tokens_per_expert_E. DeepEP returns the number
        # of global routed tokens for every local expert using other inputs.
        del num_local_tokens_per_expert_E
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()

        from torchtitan.distributed.deepep.deepep import dispatch_tokens

        hidden_states_RD, num_global_tokens_per_local_expert_e, state = dispatch_tokens(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            num_local_experts,
            self.num_experts,
            ep_group,
            score_before_experts=self.score_before_experts,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states_RD, num_global_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
    ) -> torch.Tensor:
        """Combine tokens via DeepEP.

        When sp_size == 1, combine is async — sync_combine() is deferred
        to MoE.forward, enabling overlap with shared_experts.
        When sp_size > 1, there is no overlap: sync is forced here because
        the SP expansion must read the combine result before returning.
        """
        from torchtitan.distributed.deepep.deepep import combine_tokens, sync_combine

        # pyrefly: ignore [bad-argument-type]
        combined_TD = combine_tokens(routed_output_RD, metadata.state)

        if self.sp_size > 1:
            sync_combine()
            out_TD = torch.zeros(
                num_local_tokens_after_padding * self.sp_size,
                combined_TD.shape[-1],
                device=combined_TD.device,
                dtype=combined_TD.dtype,
            )
            offset = num_local_tokens_after_padding * self.sp_rank
            out_TD[offset : offset + combined_TD.shape[0]] = combined_TD
            return out_TD

        return combined_TD


class HybridEPTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher using HybridEP for efficient token dispatch/combine.

    Uses HybridEP library kernels (GB200/NVLink72) instead of standard
    all-to-all collectives.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        """Config for HybridEP token dispatcher.

        Args:
            non_blocking_capacity_factor: Enable non-blocking HybridEP dispatch
                with a given capacity factor.

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

        non_blocking_capacity_factor: float | None = None
        pad_multiple: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.non_blocking_capacity_factor = config.non_blocking_capacity_factor
        self.pad_multiple = config.pad_multiple
        self.ep_mesh: DeviceMesh | None = None
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt = 0

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import hybridep  # noqa: F401

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh used by HybridEP dispatch / combine.

        ``tp_mesh`` provides SP coordinates so combine can expand its output
        to full sequence length (matching AllToAll's convention).
        """
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            # pyrefly: ignore [bad-assignment]
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        # Ignore input num_local_tokens_per_expert_E. HybridEP returns the
        # number of global routed tokens for every local expert using other inputs.
        del num_local_tokens_per_expert_E
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()

        from torchtitan.distributed.deepep.hybridep import dispatch_tokens

        hidden_states_RD, num_global_tokens_per_local_expert_e, state = dispatch_tokens(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            num_local_experts,
            self.num_experts,
            ep_group,
            score_before_experts=self.score_before_experts,
            non_blocking_expert_capacity_factor=self.non_blocking_capacity_factor,
            pad_multiple=self.pad_multiple,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states_RD, num_global_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
    ) -> torch.Tensor:
        """Combine tokens via HybridEP."""
        from torchtitan.distributed.deepep import hybridep

        combined_TD = hybridep.combine_tokens(
            routed_output_RD,
            metadata.state,  # pyrefly: ignore [bad-argument-type]
            pad_multiple=self.pad_multiple,
        )

        if self.sp_size > 1:
            out_TD = torch.zeros(
                num_local_tokens_after_padding * self.sp_size,
                combined_TD.shape[-1],
                device=combined_TD.device,
                dtype=combined_TD.dtype,
            )
            offset = num_local_tokens_after_padding * self.sp_rank
            out_TD[offset : offset + combined_TD.shape[0]] = combined_TD
            return out_TD

        return combined_TD
