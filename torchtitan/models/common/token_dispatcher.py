# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DeviceMesh

import spmd_types as spmd
from torchtitan.config import Configurable
from torchtitan.distributed.spmd_state import current_mesh, set_current_mesh
from torchtitan.ops.scatter_add import deterministic_scatter_add


def _local_type_like(tensor: torch.Tensor) -> spmd.LocalSpmdType:
    return dict(spmd.get_local_type(tensor)) if spmd.has_local_type(tensor) else {}


def _sp_rank_scalar(sp_rank: int | torch.SymInt) -> int | torch.SymInt | spmd.Scalar:
    if not spmd.is_type_checking():
        return sp_rank
    mesh_axis_names = spmd.current_mesh_names()
    assert mesh_axis_names is not None
    local_type = {axis: spmd.R for axis in mesh_axis_names.values()}
    if "tp" in mesh_axis_names:
        local_type[mesh_axis_names["tp"]] = spmd.V
    return spmd.Scalar(sp_rank, local_type)


def _deterministic_scatter_add_with_spmd_asserts(
    out: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    *,
    ep_enabled: bool,
) -> torch.Tensor:
    out_type = _local_type_like(out)
    mesh_axis_names = spmd.current_mesh_names() or {}
    if "tp" in mesh_axis_names:
        tp_axis = mesh_axis_names["tp"]
        if spmd.is_type_checking():
            mesh = current_mesh()
            assert mesh is not None
            pg = mesh.get_group("tp")
            out_axis_type = spmd.get_local_type(out).get(tp_axis, spmd.R)
            if out_axis_type is not spmd.P:
                out = spmd.reinterpret(
                    out,
                    pg,
                    src=out_axis_type,
                    dst=spmd.P,
                    expert_mode=True,
                )
            src_axis_type = spmd.get_local_type(src).get(tp_axis, spmd.R)
            if not ep_enabled and src_axis_type is spmd.V:
                src = spmd.reinterpret(
                    src,
                    pg,
                    src=spmd.V,
                    dst=spmd.P,
                    expert_mode=True,
                )
        # TODO: Give deterministic_scatter_add a local SPMD typing rule.
        # The scatter indices are local routing metadata, so the global
        # checker cannot currently type this custom op directly.
        with spmd.typecheck(local=True):
            out_axis_type = spmd.get_local_type(out).get(tp_axis, spmd.R)
            assert out_axis_type in (spmd.R, spmd.P)
            spmd.assert_type(index, {tp_axis: spmd.V if ep_enabled else spmd.R})
            spmd.assert_type(src, {tp_axis: spmd.V if ep_enabled else spmd.P})
        out_type[tp_axis] = spmd.P

    partition_spec = spmd.get_partition_spec(out) if spmd.has_local_type(out) else None
    with spmd.no_typecheck():
        result = deterministic_scatter_add(out, index, src)
    if spmd.is_type_checking():
        spmd.assert_type(result, out_type, partition_spec=partition_spec)
    return result


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
    # Pre-pad token count. dispatch() rounds bs*slen up to a multiple of
    # sp_size when sp_size > 1; combine() slices pad rows off using this.
    original_num_tokens: int


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
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder tokens by expert assignment for local expert computation.

        Groups tokens by expert index via histc + argsort, optionally
        applies routing scores (when ``score_before_experts`` is True).

        Args:
            x: (num_tokens, dim) input tokens
            top_scores: (num_tokens, top_k) routing scores
            selected_experts_indices: (num_tokens, top_k) expert indices

        Returns:
            routed_input: (num_tokens * top_k, dim) tokens in expert-sorted
                order, score-weighted if ``score_before_experts``
            num_tokens_per_expert: (num_experts,) token counts per expert
            token_indices_experts_sorted: (num_tokens * top_k,) token-to-original mapping
            top_scores_experts_sorted: (num_tokens * top_k,) scores in expert-sorted order
        """
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

        return (
            routed_input,
            num_tokens_per_expert,
            token_indices_experts_sorted,
            top_scores_experts_sorted,
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

        if spmd.is_type_checking():
            ep_enabled = getattr(self, "ep_mesh", None) is not None
            token_type = _local_type_like(x)
            spmd.assert_type(routed_input, token_type)
            spmd.assert_type(num_tokens_per_expert, token_type)
            if ep_enabled:
                index_type = _local_type_like(selected_experts_indices)
                score_type = _local_type_like(top_scores)
            else:
                index_type = token_type
                score_type = token_type
            spmd.assert_type(token_indices_experts_sorted, index_type)
            spmd.assert_type(top_scores_experts_sorted, score_type)

        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted=token_indices_experts_sorted,
            top_scores_experts_sorted=top_scores_experts_sorted,
        )
        return routed_input, num_tokens_per_expert, metadata

    def _scatter_local_expert_outputs(
        self,
        out: torch.Tensor,
        routed_output: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
        top_scores_experts_sorted: torch.Tensor,
    ) -> torch.Tensor:
        routed_type = _local_type_like(routed_output)
        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(routed_output.dtype)
            if spmd.is_type_checking():
                spmd.assert_type(routed_output, routed_type)

        ep_enabled = getattr(self, "ep_mesh", None) is not None
        index = token_indices_experts_sorted.reshape(-1, 1).expand(-1, out.shape[-1])
        return _deterministic_scatter_add_with_spmd_asserts(
            out,
            index,
            routed_output,
            ep_enabled=ep_enabled,
        )

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
        out = torch.zeros_like(x)
        return self._scatter_local_expert_outputs(
            out,
            routed_output,
            metadata.token_indices_experts_sorted,
            metadata.top_scores_experts_sorted,
        )


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
        self.sparse_mesh: DeviceMesh | None = None
        # Sequence-parallel split coordinates derived from tp_mesh.
        # ``sp_rank`` uses ``DeviceMesh._sym_get_coordinate`` so it is a
        # ``SymInt`` under CooR precompile, keeping the FX graph
        # rank-agnostic. Defaults are the TP=1 values.
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt | spmd.Scalar = 0

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
            self.set_sp_rank(tp_mesh._sym_get_coordinate(0))

    def set_sp_rank(self, sp_rank: int | torch.SymInt) -> None:
        self.sp_rank = _sp_rank_scalar(sp_rank)

    def _split_along_sp(self, *tensors: torch.Tensor) -> list[torch.Tensor]:
        """Split tensors along the first dim across TP ranks for sequence parallel."""
        sp_size = self.sp_size
        sp_rank = self.sp_rank
        results = []
        for tensor in tensors:
            assert tensor.is_contiguous()
            num_tokens = tensor.shape[0]
            if num_tokens % sp_size != 0:
                raise ValueError(
                    "Uneven split of tokens is not supported yet. "
                    "Requires TP degree dividing batch size * seq len."
                )
            local_num_tokens = num_tokens // sp_size
            offset = sp_rank * local_num_tokens
            local_tensor = torch.narrow(tensor, 0, offset, local_num_tokens)
            if spmd.is_type_checking():
                expected_type = self._sp_shard_type(tensor)
                mesh_axis_names = spmd.current_mesh_names() or {}
                if "tp" in mesh_axis_names:
                    tp_axis = mesh_axis_names["tp"]
                    actual_tp = spmd.get_local_type(local_tensor).get(tp_axis, spmd.R)
                    expected_tp = expected_type[tp_axis]
                    if actual_tp is not expected_tp:
                        mesh = current_mesh()
                        assert mesh is not None
                        local_tensor = spmd.reinterpret(
                            local_tensor,
                            mesh.get_group("tp"),
                            src=actual_tp,
                            dst=expected_tp,
                            expert_mode=True,
                        )
                spmd.assert_type(local_tensor, expected_type)
            results.append(local_tensor)
        return results

    def _sp_shard_type(self, tensor: torch.Tensor) -> spmd.LocalSpmdType:
        local_type = _local_type_like(tensor)
        mesh_axis_names = spmd.current_mesh_names() or {}
        if "tp" in mesh_axis_names:
            local_type[mesh_axis_names["tp"]] = spmd.V
        return local_type

    def _sparse_token_placement(
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

    def _dense_token_placement(self) -> dict[str, spmd.PerMeshAxisSpmdType]:
        mesh_axis_names = spmd.current_mesh_names() or {}
        placement: dict[str, spmd.PerMeshAxisSpmdType] = {}
        for axis_name in ("dp", "cp"):
            if axis_name in mesh_axis_names:
                placement[axis_name] = spmd.V
        if "tp" in mesh_axis_names:
            placement["tp"] = spmd.V if self.sp_size > 1 else spmd.R
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

        When ep_mesh is None (EP=1), falls back to local dispatch — no
        all-to-all communication, just local token reordering with padding.

        When sp_size > 1, inputs are first split along the token dim so each
        TP rank processes a disjoint subset.

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
        if self.ep_mesh is None:
            return super().dispatch(x, top_scores, selected_experts_indices)
        assert self.sparse_mesh is not None

        ep_size = self.ep_mesh.size()
        original_num_tokens = x.shape[0]

        if self.sp_size > 1:
            assert self.sp_rank >= 0, (
                "sp_rank must be set before use. "
                "GroupedExperts.parallelize() should set it from "
                "tp_mesh._sym_get_coordinate()."
            )
            pad = (-original_num_tokens) % self.sp_size
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad))
                top_scores = F.pad(top_scores, (0, 0, 0, pad))
                selected_experts_indices = F.pad(
                    selected_experts_indices, (0, 0, 0, pad)
                )
            x, top_scores, selected_experts_indices = self._split_along_sp(
                x, top_scores, selected_experts_indices
            )

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
                    self._sparse_token_placement(),
                )
                spmd.assert_type(
                    routed_input,
                    self._sparse_token_placement(ep=spmd.S(0)),
                )
                spmd.assert_type(
                    num_tokens_per_expert_group,
                    self._sparse_token_placement(ep=spmd.S(0)),
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
            token_indices_experts_sorted=token_indices_experts_sorted,
            top_scores_experts_sorted=top_scores_experts_sorted,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
            original_num_tokens=original_num_tokens,
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
        if spmd.is_type_checking():
            spmd.assert_type(routed_input, _local_type_like(routed_input))
            spmd.assert_type(permuted_indices, _local_type_like(routed_input))
            spmd.assert_type(
                num_tokens_per_expert,
                _local_type_like(num_tokens_per_expert_group),
        )
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
            spmd.assert_type(out_unpermuted, _local_type_like(routed_output))
        return out_unpermuted

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + score + scatter_add.

        When sp_size > 1, dispatch uses local token indices.
        Combine offsets them to global positions so scatter_add
        into full x is correct.

        Args:
            routed_output: (R, dim) expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x: (num_tokens, dim) original input tokens

        Returns:
            (num_tokens, dim) combined output.
        """
        # EP=1: fall back to local combine (no all-to-all needed)
        if self.ep_mesh is None:
            return super().combine(routed_output, metadata, x)
        assert self.sparse_mesh is not None

        with set_current_mesh(self.sparse_mesh):
            routed_output = self._unpermute(
                routed_output, metadata.input_shape, metadata.permuted_indices
            )
            ep_pg = self.sparse_mesh.get_group("ep")
            spmd.assert_type(routed_output, self._sparse_token_placement(ep=spmd.S(0)))
            routed_output = spmd.all_to_all(
                routed_output,
                ep_pg,
                src=spmd.S(0),
                dst=spmd.S(0),
                output_split_sizes=metadata.input_splits,
                input_split_sizes=metadata.output_splits,
            )

        if spmd.is_type_checking():
            routed_output = spmd.reinterpret_mesh(
                routed_output,
                self._dense_token_placement(),
            )

        out = torch.zeros_like(x)
        routed_type = _local_type_like(routed_output)

        # With SP, token indices are 0-based within the local shard.
        # Offset to global positions for the full-size scatter buffer.
        token_indices_experts_sorted = metadata.token_indices_experts_sorted
        if self.sp_size > 1:
            padded_num_tokens = metadata.original_num_tokens + (
                (-metadata.original_num_tokens) % self.sp_size
            )
            local_num_tokens = padded_num_tokens // self.sp_size
            token_indices_experts_sorted = (
                token_indices_experts_sorted + local_num_tokens * self.sp_rank
            )
            assert isinstance(token_indices_experts_sorted, torch.Tensor)
            if padded_num_tokens != metadata.original_num_tokens:
                mask = token_indices_experts_sorted < metadata.original_num_tokens
                token_indices_experts_sorted = token_indices_experts_sorted[mask]
                routed_output = routed_output[mask]
        if spmd.is_type_checking():
            spmd.assert_type(routed_output, routed_type)
        return self._scatter_local_expert_outputs(
            out,
            routed_output,
            token_indices_experts_sorted,
            metadata.top_scores_experts_sorted,
        )


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

    def dispatch(self, x, top_scores, selected_experts_indices):
        if self.ep_mesh is None:
            raise ValueError(
                "TorchAOTokenDispatcher requires expert parallelism (ep_mesh must be set). "
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
        self.ep_mesh: DeviceMesh | None = None
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt | spmd.Scalar = 0

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
            self.sp_rank = _sp_rank_scalar(tp_mesh._sym_get_coordinate(0))

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
        self.ep_mesh: DeviceMesh | None = None
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt | spmd.Scalar = 0

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
            self.sp_rank = _sp_rank_scalar(tp_mesh._sym_get_coordinate(0))

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
