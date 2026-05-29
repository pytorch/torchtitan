# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DeviceMesh

import spmd_types as spmd
from torchtitan.config import Configurable
from torchtitan.distributed.spmd_state import current_mesh, set_current_mesh
from torchtitan.ops.scatter_add import deterministic_scatter_add
from torchtitan.protocols.module import named_placement_to_spmd
from torchtitan.protocols.types import MeshAxisName


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
        self.sparse_mesh: DeviceMesh | None = None

    @property
    def sp_size(self) -> int:
        mesh = current_mesh()
        if (
            mesh is None
            or mesh.mesh_dim_names is None
            or "tp" not in mesh.mesh_dim_names
        ):
            return 1
        tp_axis = mesh.mesh_dim_names.index("tp")
        return mesh.size(tp_axis)

    @property
    def sp_rank(self) -> int | torch.SymInt | spmd.Scalar:
        mesh = current_mesh()
        if (
            mesh is None
            or mesh.mesh_dim_names is None
            or "tp" not in mesh.mesh_dim_names
        ):
            return 0
        tp_axis = mesh.mesh_dim_names.index("tp")
        sp_rank = mesh._sym_get_coordinate(tp_axis)
        if not spmd.is_type_checking():
            return sp_rank
        mesh_axis_names = spmd.current_mesh_names()
        assert mesh_axis_names is not None
        local_type = {axis: spmd.R for axis in mesh_axis_names.values()}
        local_type[mesh_axis_names["tp"]] = spmd.V
        return spmd.Scalar(sp_rank, local_type)

    def _local_reorder(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder tokens by expert assignment for local expert computation.

        Groups tokens by expert index via histc + argsort, optionally
        applies routing scores (when ``score_before_experts`` is True).

        Args:
            x_TD: ``(T, D)`` input tokens
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices

        Returns:
            routed_input_ND: ``(N, D)`` where N = T*K. Tokens in expert-sorted
                order, score-weighted if ``score_before_experts``.
            num_tokens_per_expert_E: ``(E,)`` token counts per expert
            token_indices_experts_sorted_N: ``(N,)`` token-to-original mapping
            topk_scores_experts_sorted_N: ``(N,)`` scores in expert-sorted order
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert_E = torch.histc(
            topk_expert_ids_TK.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts where N = T*K
        token_indices_experts_sorted_N = torch.argsort(
            topk_expert_ids_TK.view(-1), stable=True
        )
        topk_scores_experts_sorted_N = topk_scores_TK.view(-1)[
            token_indices_experts_sorted_N
        ]
        token_indices_experts_sorted_N = token_indices_experts_sorted_N // self.top_k
        routed_input_ND = x_TD[token_indices_experts_sorted_N]

        if self.score_before_experts:
            routed_input_ND = (
                routed_input_ND.to(torch.float32)
                * topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(x_TD.dtype)

        return (
            routed_input_ND,
            num_tokens_per_expert_E,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        )

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
        (
            routed_input,
            num_tokens_per_expert,
            token_indices_experts_sorted,
            top_scores_experts_sorted,
        ) = self._local_reorder(x, top_scores, selected_experts_indices)

        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted,
            topk_scores_experts_sorted_N=top_scores_experts_sorted,
        )
        return routed_input, num_tokens_per_expert, metadata

    def _scatter_expert_outputs(
        self,
        out: torch.Tensor,
        routed_output: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
    ) -> torch.Tensor:
        index = token_indices_experts_sorted.reshape(-1, 1).expand(-1, out.shape[-1])

        with spmd.no_typecheck():
            result = deterministic_scatter_add(out, index, routed_output)
        if spmd.is_type_checking():
            spmd.assert_type(
                result,
                named_placement_to_spmd(
                    {
                        MeshAxisName.DP: spmd.V,
                        MeshAxisName.CP: spmd.V,
                        MeshAxisName.TP: spmd.P,
                    }
                ),
            )
        return result

    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Score and scatter_add routed expert outputs.

        Args:
            routed_output: (num_tokens * top_k, dim) expert outputs
            metadata: LocalDispatchMetadata from dispatch()
            x: (num_tokens, dim) original input tokens

        Returns:
            (num_tokens, dim) combined output.
        """
        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(routed_output.dtype)

        return self._scatter_expert_outputs(
            x.new_zeros(x.shape),
            routed_output,
            metadata.token_indices_experts_sorted_N,
        )


class AllToAllTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher for EP>1. Handles token reorder + all-to-all dispatch/combine.

    Handles the full token routing lifecycle:
    dispatch (reorder + EP all-to-all) and combine (reverse).

    The owning ``GroupedExperts.parallelize`` installs ``sparse_mesh``. Dense
    TP sequence-parallel coordinates are derived from the current dense mesh.
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
        self.sparse_mesh: DeviceMesh | None = None

    def _sparse_placement(
        self,
        *,
        ep: spmd.PerMeshAxisSpmdType = spmd.V,
    ) -> dict[str, spmd.PerMeshAxisSpmdType]:
        mesh_axis_names = spmd.current_mesh_names() or {}
        placement: dict[str, spmd.PerMeshAxisSpmdType] = {}
        for axis_name in ("dp_replicate", "efsdp"):
            if axis_name in mesh_axis_names:
                placement[axis_name] = spmd.V
        if "ep" in mesh_axis_names:
            placement["ep"] = ep
        return placement

    def _dense_placement(
        self,
        *,
        tp: spmd.PerMeshAxisSpmdType,
    ) -> dict[str, spmd.PerMeshAxisSpmdType]:
        mesh_axis_names = spmd.current_mesh_names() or {}
        placement: dict[str, spmd.PerMeshAxisSpmdType] = {}
        for axis_name in ("dp", "cp"):
            if axis_name in mesh_axis_names:
                placement[axis_name] = spmd.V
        if "tp" in mesh_axis_names:
            placement["tp"] = tp
        return placement

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, AllToAllDispatchMetadata | LocalDispatchMetadata
    ]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        When ``sparse_mesh`` is None (EP=1), falls back to local dispatch — no
        all-to-all communication, just local token reordering with padding.

        Args:
            x: (num_local_tokens, dim) local token shard
            top_scores: (num_local_tokens, top_k) routing scores
            selected_experts_indices: (num_local_tokens, top_k) expert indices

        Returns:
            routed_input: (R, dim) tokens in expert-major order for local experts
            num_tokens_per_expert_local: (num_local_experts,) token counts
            metadata: dispatch metadata for combine()
        """
        # EP=1: fall back to local dispatch (no all-to-all needed)
        if self.sparse_mesh is None:
            return super().dispatch(x, top_scores, selected_experts_indices)

        ep_size = self.sparse_mesh.get_group("ep").size()
        (
            routed_input,
            num_tokens_per_expert,
            token_indices_experts_sorted,
            top_scores_experts_sorted,
        ) = self._local_reorder(x, top_scores, selected_experts_indices)

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            with set_current_mesh(self.sparse_mesh):
                with spmd.no_typecheck():
                    ep_pg = self.sparse_mesh.get_group("ep")
                    num_tokens_per_expert_group = spmd.all_to_all(
                        num_tokens_per_expert,
                        ep_pg,
                        src=spmd.S(0),
                        dst=spmd.S(0),
                    )
                if spmd.is_type_checking():
                    spmd.assert_type(
                        num_tokens_per_expert_group,
                        self._sparse_placement(ep=spmd.S(0)),
                    )
                with spmd.no_typecheck():
                    non_blocking = not torch.compiler.is_compiling()
                    input_splits = (
                        num_tokens_per_expert.view(ep_size, -1)
                        .sum(dim=1)
                        .to(torch.device("cpu"), non_blocking=non_blocking)
                    )
                    output_splits = (
                        num_tokens_per_expert_group.view(ep_size, -1)
                        .sum(dim=1)
                        .to(torch.device("cpu"), non_blocking=False)
                    )
                    input_splits_list = input_splits.tolist()
                    output_splits_list = output_splits.tolist()

        with set_current_mesh(self.sparse_mesh):
            ep_pg = self.sparse_mesh.get_group("ep")
            if spmd.is_type_checking():
                routed_input = spmd.reinterpret_mesh(
                    routed_input,
                    self._sparse_placement(),
                )
                spmd.assert_type(
                    routed_input,
                    self._sparse_placement(ep=spmd.S(0)),
                )
            routed_input = spmd.all_to_all(
                routed_input,
                ep_pg,
                src=spmd.S(0),
                dst=spmd.S(0),
                output_split_sizes=output_splits_list,
                input_split_sizes=input_splits_list,
            )

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
            token_indices_experts_sorted_N=token_indices_experts_sorted,
            topk_scores_experts_sorted_N=top_scores_experts_sorted,
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
        input_shape = routed_input.shape
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
        routed_input = routed_input[permuted_indices, :]
        return (
            input_shape,
            routed_input,
            permuted_indices,
            num_tokens_per_expert,
        )

    def _unpermute(self, routed_output, input_shape, permuted_indices):
        """Reverse expert-major reordering."""
        out_unpermuted = routed_output.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = routed_output
        if spmd.is_type_checking():
            spmd.assert_type(out_unpermuted, spmd.get_local_type(routed_output))
        return out_unpermuted

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + score + scatter_add.

        With SP, x is the local shard. Combine scatters into a full-size
        buffer so routed results from all SP ranks occupy global positions.

        Args:
            routed_output: (R, dim) expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x: (num_tokens, dim) original input tokens

        Returns:
            (num_tokens, dim) combined output.
        """
        # EP=1: fall back to local combine (no all-to-all needed)
        if self.sparse_mesh is None:
            return super().combine(
                routed_output,
                metadata,
                x,
            )

        with set_current_mesh(self.sparse_mesh):
            routed_output = self._unpermute(
                routed_output, metadata.input_shape, metadata.permuted_indices
            )
            ep_pg = self.sparse_mesh.get_group("ep")
            spmd.assert_type(routed_output, self._sparse_placement(ep=spmd.S(0)))
            routed_output = spmd.all_to_all(
                routed_output,
                ep_pg,
                src=spmd.S(0),
                dst=spmd.S(0),
                output_split_sizes=metadata.input_splits,
                input_split_sizes=metadata.output_splits,
            )

        if spmd.is_type_checking():
            dense_placement = self._dense_placement(tp=spmd.V)
            routed_output = spmd.reinterpret_mesh(routed_output, dense_placement)

        out = torch.zeros(
            x.shape[0] * self.sp_size,
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(routed_output.dtype)

        if self.sp_size > 1:
            token_indices_experts_sorted = (
                metadata.token_indices_experts_sorted_N + x.shape[0] * self.sp_rank
            )
        else:
            token_indices_experts_sorted = metadata.token_indices_experts_sorted_N

        return self._scatter_expert_outputs(
            out,
            routed_output,
            token_indices_experts_sorted,
        )


class TorchAOTokenDispatcher(AllToAllTokenDispatcher):
    """Token dispatcher with token group padding for quantized grouped GEMMs.

    Uses torchao's ``permute_and_pad`` instead of the standard ``_permute`` to
    reorder tokens into expert-major order and pad each expert's token group to
    a multiple of ``pad_multiple``. This alignment is required by FP8/MXFP8
    quantized grouped GEMM kernels (e.g. 16 for FP8, 32 for MXFP8).

    Requires EP to be enabled (sparse_mesh must be set). Raises ValueError
    if sparse_mesh is None, since quantized grouped GEMMs need padded token
    groups which are only produced by the EP permute_and_pad path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pad_multiple: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.pad_multiple = config.pad_multiple

    def dispatch(self, x, top_scores, selected_experts_indices):
        if self.sparse_mesh is None:
            raise ValueError(
                "TorchAOTokenDispatcher requires expert parallelism (sparse_mesh must be set). "
                "Quantized grouped GEMMs need padded token groups, which requires EP>1. "
            )
        return super().dispatch(x, top_scores, selected_experts_indices)

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

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import deepep  # noqa: F401

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        assert self.sparse_mesh is not None, (
            "sparse_mesh must be set before dispatch. "
            "GroupedExperts.parallelize() should set it."
        )
        ep_group = self.sparse_mesh.get_group("ep")
        num_local_experts = self.num_experts // ep_group.size()

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
    ) -> torch.Tensor:
        """Combine tokens via DeepEP.

        When sp_size == 1, combine is async — sync_combine() is deferred
        to MoE.forward, enabling overlap with shared_experts.
        When sp_size > 1, there is no overlap: sync is forced here because
        the SP expansion must read the combine result before returning.
        """
        from torchtitan.distributed.deepep.deepep import combine_tokens, sync_combine

        # pyrefly: ignore [bad-argument-type]
        routed_output = combine_tokens(routed_output, metadata.state)

        if self.sp_size > 1:
            sync_combine()
            out = torch.zeros(
                routed_output.shape[0] * self.sp_size,
                routed_output.shape[-1],
                device=routed_output.device,
                dtype=routed_output.dtype,
            )
            offset = routed_output.shape[0] * self.sp_rank
            out[offset : offset + routed_output.shape[0]] = routed_output
            return out

        return routed_output


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

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import hybridep  # noqa: F401

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        assert self.sparse_mesh is not None, (
            "sparse_mesh must be set before dispatch. "
            "GroupedExperts.parallelize() should set it."
        )
        ep_group = self.sparse_mesh.get_group("ep")
        num_local_experts = self.num_experts // ep_group.size()

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

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states, tokens_per_expert, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Combine tokens via HybridEP."""
        from torchtitan.distributed.deepep import hybridep

        routed_output = hybridep.combine_tokens(
            routed_output,
            metadata.state,  # pyrefly: ignore [bad-argument-type]
            pad_multiple=self.pad_multiple,
        )

        if self.sp_size > 1:
            out = torch.zeros(
                routed_output.shape[0] * self.sp_size,
                routed_output.shape[-1],
                device=routed_output.device,
                dtype=routed_output.dtype,
            )
            offset = routed_output.shape[0] * self.sp_rank
            out[offset : offset + routed_output.shape[0]] = routed_output
            return out

        return routed_output
