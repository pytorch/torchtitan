# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
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
        x_ND: torch.Tensor,
        top_scores_NK: torch.Tensor,
        selected_experts_indices_NK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder tokens by expert assignment for local expert computation.

        Groups tokens by expert index via histc + argsort, optionally
        applies routing scores (when ``score_before_experts`` is True).

        Returns:
            routed_input_TD, num_tokens_per_expert_E,
            token_indices_experts_sorted_T, top_scores_experts_sorted_T
        """
        num_tokens_per_expert_E = torch.histc(
            selected_experts_indices_NK.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        token_indices_experts_sorted_T = torch.argsort(
            selected_experts_indices_NK.view(-1), stable=True
        )

        top_scores_experts_sorted_T = top_scores_NK.view(-1)[
            token_indices_experts_sorted_T
        ]
        token_indices_experts_sorted_T = (
            token_indices_experts_sorted_T // self.top_k
        )

        routed_input_TD = x_ND[token_indices_experts_sorted_T]

        if self.score_before_experts:
            routed_input_TD = (
                routed_input_TD.to(torch.float32)
                * top_scores_experts_sorted_T.reshape(-1, 1)
            ).to(x_ND.dtype)

        return (
            routed_input_TD,
            num_tokens_per_expert_E,
            token_indices_experts_sorted_T,
            top_scores_experts_sorted_T,
        )

    def dispatch(
        self,
        x_ND: torch.Tensor,
        top_scores_NK: torch.Tensor,
        selected_experts_indices_NK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert assignment for local expert computation.

        Returns:
            routed_input_TD, num_tokens_per_expert_E, metadata
        """
        (
            routed_input_TD,
            num_tokens_per_expert_E,
            token_indices_experts_sorted_T,
            top_scores_experts_sorted_T,
        ) = self._local_reorder(
            x_ND, top_scores_NK, selected_experts_indices_NK
        )

        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted=token_indices_experts_sorted_T,
            top_scores_experts_sorted=top_scores_experts_sorted_T,
        )
        return routed_input_TD, num_tokens_per_expert_E, metadata

    def combine(
        self,
        routed_output_TD: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x_ND: torch.Tensor,
    ) -> torch.Tensor:
        """Score and scatter_add routed expert outputs.

        Returns:
            out_ND: ``(N, D)`` combined output.
        """
        out_ND = torch.zeros_like(x_ND)

        if not self.score_before_experts:
            routed_output_TD = (
                routed_output_TD.to(torch.float32)
                * metadata.top_scores_experts_sorted.reshape(-1, 1)
            ).to(routed_output_TD.dtype)

        dim = x_ND.shape[-1]
        out_ND = deterministic_scatter_add(
            out_ND,
            metadata.token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim),
            routed_output_TD,
        )
        return out_ND


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
        x_ND: torch.Tensor,
        top_scores_NK: torch.Tensor,
        selected_experts_indices_NK: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, AllToAllDispatchMetadata | LocalDispatchMetadata
    ]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        When ep_mesh is None (EP=1), falls back to local dispatch.

        Returns:
            routed_input_TD, num_tokens_per_expert_E, metadata
        """
        if self.ep_mesh is None:
            return super().dispatch(
                x_ND, top_scores_NK, selected_experts_indices_NK
            )

        ep_size = self.ep_mesh.size()

        (
            routed_input_TD,
            num_tokens_per_expert_E,
            token_indices_experts_sorted_T,
            top_scores_experts_sorted_T,
        ) = self._local_reorder(
            x_ND, top_scores_NK, selected_experts_indices_NK
        )

        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert_E,
                None,
                None,
                group=self.ep_mesh,
            )
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            non_blocking = not torch.compiler.is_compiling()
            input_splits = (
                num_tokens_per_expert_E.view(ep_size, -1)
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

        routed_input_TD = all_to_all_single_autograd(
            routed_input_TD,
            output_splits_list,
            input_splits_list,
            self.ep_mesh,
        )

        # Reorder from rank-major to expert-major via _permute.
        num_local_experts = num_tokens_per_expert_group.shape[0] // ep_size
        (
            input_shape,
            routed_input_TD,
            permuted_indices,
            num_tokens_per_expert_E,
        ) = self._permute(
            routed_input_TD,
            num_tokens_per_expert_group,
            ep_size,
            num_local_experts,
        )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted=token_indices_experts_sorted_T,
            top_scores_experts_sorted=top_scores_experts_sorted_T,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        return routed_input_TD, num_tokens_per_expert_E, metadata

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
        routed_output_TD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_ND: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + score + scatter_add.

        Returns:
            out_ND: ``(N, D)`` combined output (N may include SP expansion).
        """
        if self.ep_mesh is None:
            return super().combine(routed_output_TD, metadata, x_ND)

        routed_output_TD = self._unpermute(
            routed_output_TD, metadata.input_shape, metadata.permuted_indices
        )

        routed_output_TD = all_to_all_single_autograd(
            routed_output_TD,
            metadata.input_splits,
            metadata.output_splits,
            self.ep_mesh,
        )

        out_ND = torch.zeros(
            x_ND.shape[0] * self.sp_size,
            x_ND.shape[-1],
            device=x_ND.device,
            dtype=x_ND.dtype,
        )

        if not self.score_before_experts:
            routed_output_TD = (
                routed_output_TD.to(torch.float32)
                * metadata.top_scores_experts_sorted.reshape(-1, 1)
            ).to(routed_output_TD.dtype)

        if self.sp_size > 1:
            token_indices_experts_sorted_T = (
                metadata.token_indices_experts_sorted
                + x_ND.shape[0] * self.sp_rank
            )
        else:
            token_indices_experts_sorted_T = (
                metadata.token_indices_experts_sorted
            )

        assert isinstance(token_indices_experts_sorted_T, torch.Tensor)
        out_ND = deterministic_scatter_add(
            out_ND,
            token_indices_experts_sorted_T.reshape(-1, 1).expand(
                -1, out_ND.shape[-1]
            ),
            routed_output_TD,
        )
        return out_ND


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

    def dispatch(self, x_ND, top_scores_NK, selected_experts_indices_NK):
        if self.ep_mesh is None:
            raise ValueError(
                "TorchAOTokenDispatcher requires expert parallelism (ep_mesh must be set). "
                "Quantized grouped GEMMs need padded token groups, which requires EP>1. "
            )
        return super().dispatch(
            x_ND, top_scores_NK, selected_experts_indices_NK
        )

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
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_ND: torch.Tensor,
        top_scores_NK: torch.Tensor,
        selected_experts_indices_NK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()

        from torchtitan.distributed.deepep.deepep import dispatch_tokens

        hidden_states_TD, tokens_per_expert_E, state = dispatch_tokens(
            x_ND,
            selected_experts_indices_NK,
            top_scores_NK,
            num_local_experts,
            self.num_experts,
            ep_group,
            score_before_experts=self.score_before_experts,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states_TD, tokens_per_expert_E, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_TD: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x_ND: torch.Tensor,
    ) -> torch.Tensor:
        """Combine tokens via DeepEP.

        When sp_size == 1, combine is async — sync_combine() is deferred
        to MoE.forward, enabling overlap with shared_experts.
        When sp_size > 1, there is no overlap: sync is forced here because
        the SP expansion must read the combine result before returning.
        """
        from torchtitan.distributed.deepep.deepep import combine_tokens, sync_combine

        # pyrefly: ignore [bad-argument-type]
        combined_ND = combine_tokens(routed_output_TD, metadata.state)

        if self.sp_size > 1:
            sync_combine()
            out_ND = torch.zeros(
                combined_ND.shape[0] * self.sp_size,
                combined_ND.shape[-1],
                device=combined_ND.device,
                dtype=combined_ND.dtype,
            )
            offset = combined_ND.shape[0] * self.sp_rank
            out_ND[offset : offset + combined_ND.shape[0]] = combined_ND
            return out_ND

        return combined_ND


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
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_ND: torch.Tensor,
        top_scores_NK: torch.Tensor,
        selected_experts_indices_NK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()

        from torchtitan.distributed.deepep.hybridep import dispatch_tokens

        hidden_states_TD, tokens_per_expert_E, state = dispatch_tokens(
            x_ND,
            selected_experts_indices_NK,
            top_scores_NK,
            num_local_experts,
            self.num_experts,
            ep_group,
            score_before_experts=self.score_before_experts,
            non_blocking_expert_capacity_factor=self.non_blocking_capacity_factor,
            pad_multiple=self.pad_multiple,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states_TD, tokens_per_expert_E, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_TD: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x_ND: torch.Tensor,
    ) -> torch.Tensor:
        """Combine tokens via HybridEP."""
        from torchtitan.distributed.deepep import hybridep

        combined_ND = hybridep.combine_tokens(
            routed_output_TD,
            metadata.state,  # pyrefly: ignore [bad-argument-type]
            pad_multiple=self.pad_multiple,
        )

        if self.sp_size > 1:
            out_ND = torch.zeros(
                combined_ND.shape[0] * self.sp_size,
                combined_ND.shape[-1],
                device=combined_ND.device,
                dtype=combined_ND.dtype,
            )
            offset = combined_ND.shape[0] * self.sp_rank
            out_ND[offset : offset + combined_ND.shape[0]] = combined_ND
            return out_ND

        return combined_ND
