# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compiler passes for graph_trainer training.

This module provides pass orchestration: building the pass list, applying passes
in order, and the pass registries.  Individual passes live in dedicated modules:

- ``memory_policy.py`` — SAC tagging and memory policy dispatch
- ``inductor_passes.py`` — regional and full Inductor compilation
- ``cudagraph.py`` — cudagraph wrapping and kernel annotations
- ``fsdp_passes.py`` — FSDP bucketing and resharding
- ``remove_noop_passes.py`` — no-op removal (detach, identity view/slice)
- ``performance_passes.py`` — opt-in numerics-changing optimizations
- ``selective_activation_remat.py`` — activation rematerialization
- ``cpu_offload.py`` — CPU offload insertion
- ``custom_codegen.py`` — custom code generation for profiling/debugging
"""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import Callable

import torch
from torch._logging import trace_structured

from torchtitan.experiments.graph_trainer.cpu_offload import apply_cpu_offload_pass
from torchtitan.experiments.graph_trainer.cudagraph import (
    cudagraph_pass,
    insert_kernel_annotations_pass,
)
from torchtitan.experiments.graph_trainer.custom_codegen import custom_codegen_pass
from torchtitan.experiments.graph_trainer.debug_utils import (
    log_graph_diff,
    snapshot_graph,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    overlap_fsdp_ag_rs_pass,
)
from torchtitan.experiments.graph_trainer.inductor_passes import (
    annotate_flex_attention_for_regional_inductor_pass,
    full_inductor_compilation_pass,
    regional_inductor_pass,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.experiments.graph_trainer.memory_policy import (
    tag_with_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_detach_pass,
    remove_identity_slice_pass,
    remove_identity_view_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.tools.logging import logger

aten = torch.ops.aten
c10d = torch.ops._c10d_functional


def normalize_view_ops_as_reshape(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Replace aten.view and aten._unsafe_view with aten.reshape.

    Downstream passes expect aten.reshape.default for pattern matching.
    """
    view_targets = {aten.view.default, aten._unsafe_view.default}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in view_targets:
            node.target = aten.reshape.default
    gm.graph.lint()
    gm.recompile()
    return gm


def remove_redundant_contiguous_clone_pass(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Remove aten.clone(..., memory_format=contiguous_format) calls whose
    input is already contiguous (per FakeTensor stride metadata).

    FSDP weight-unpack patterns (split_with_sizes -> view.dtype -> clone) often
    produce a clone whose input already has contiguous strides into the bucket
    buffer; the clone(contiguous_format) is then a redundant memcpy. Running
    this before ``auto_overlap_bucketing_pass`` lets the bucketer see fewer ops.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    def _is_contig(t: FakeTensor) -> bool:
        shape = list(t.size())
        strides = list(t.stride())
        expected: list = []
        s = 1
        for d in reversed(shape):
            expected.append(s)
            s *= d
        return strides == list(reversed(expected))

    removed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is not aten.clone.default:
            continue
        # Default ``memory_format`` for ``aten.clone`` is contiguous_format;
        # treat both the explicit and the implicit case as redundant when the
        # input already has contiguous strides.
        mem_fmt = node.kwargs.get("memory_format", torch.contiguous_format)
        if mem_fmt is not torch.contiguous_format:
            continue
        if not node.args or not isinstance(node.args[0], torch.fx.Node):
            continue
        in_node = node.args[0]
        in_val = in_node.meta.get("val")
        if not isinstance(in_val, FakeTensor):
            continue
        if not _is_contig(in_val):
            continue
        # Clone is redundant; route users to the input directly.
        node.replace_all_uses_with(in_node)
        gm.graph.erase_node(node)
        removed += 1
    if removed > 0:
        logger.info(f"Removed {removed} redundant clone(contiguous_format) calls")
    gm.graph.lint()
    gm.recompile()
    return gm


def auto_overlap_bucketing_pass(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Auto-schedule FSDP collectives for overlap (replaces manual bucketing).

    Uses upstream ``schedule_overlap_bucketing`` which estimates collective
    and compute runtimes from a roofline model and picks bucket sizes
    that saturate NCCL bandwidth, then reorders to maximize overlap.
    Respects a memory budget so peak memory doesn't blow up.
    """
    from torch._inductor.fx_passes.overlap_scheduling import (
        schedule_overlap_bucketing,
    )

    schedule_overlap_bucketing(
        gm,
        collective_bucketing=True,
        max_memory_increase_gb=2.0,
        max_memory_increase_ratio=0.05,
    )
    gm.recompile()
    return gm


def fuse_mm_reduce_scatter_pass(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Custom matcher for mm -> {view|reshape|_to_copy|permute|transpose|_unsafe_view}* -> RS.

    Replaces the chain with ``symm_mem.fused_matmul_reduce_scatter`` so the
    matmul can be chunked and pipelined with the reduce-scatter over NVLink
    symmetric memory. Looser than upstream ``micro_pipeline_tp_pass``'s
    ``_find_producer_matmul`` which only whitelists ``reshape -> mm -> reshape``
    (both sides) plus a bare ``mm``. Our graph has additional
    ``_to_copy``/``permute``/``transpose`` ops between mm and the collective
    on this Llama3 8B FSDP=4 x TP=2 configuration.
    """
    from torch._inductor.fx_passes.overlap_scheduling import get_group_name
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    # Force registration of the fused_matmul_reduce_scatter op (and friends).
    # Importing the symmetric_memory module registers the ops with the
    # ``symm_mem`` namespace; otherwise ``torch.ops.symm_mem.fused_matmul_reduce_scatter``
    # raises AttributeError.
    import torch.distributed._symmetric_memory  # noqa: F401

    symm_mem = torch.ops.symm_mem

    traversable_targets = {
        aten.view.default,
        aten.reshape.default,
        aten._to_copy.default,
        aten.permute.default,
        aten.transpose.int,
        aten.t.default,
        aten._unsafe_view.default,
    }

    def _trace_back_to_mm(node: torch.fx.Node) -> torch.fx.Node | None:
        """Walk single-input ops backwards until we hit an mm, or fail."""
        cur: torch.fx.Node | None = node
        while cur is not None and cur.target in traversable_targets:
            if not cur.args or not isinstance(cur.args[0], torch.fx.Node):
                return None
            cur = cur.args[0]
        if cur is not None and cur.target is aten.mm.default:
            return cur
        return None

    def _decode_split_cat(
        cat_node: torch.fx.Node,
    ) -> tuple[torch.fx.Node, int] | None:
        """If ``cat_node`` is the result of cat(split(input, ..., dim=k)), return
        (input, k). Otherwise return None.
        """
        if cat_node.target is not aten.cat.default:
            return None
        if not cat_node.args:
            return None
        cat_inputs = cat_node.args[0]
        if not isinstance(cat_inputs, (list, tuple)) or not cat_inputs:
            return None
        # All items should be getitem(split_node, idx).
        split_node = None
        seen_indices: set[int] = set()
        for item in cat_inputs:
            if not isinstance(item, torch.fx.Node):
                return None
            if item.target is not __import__("operator").getitem:
                return None
            if not item.args or not isinstance(item.args[0], torch.fx.Node):
                return None
            sn = item.args[0]
            if split_node is None:
                split_node = sn
            elif sn is not split_node:
                return None
            seen_indices.add(int(item.args[1]))
        if split_node is None:
            return None
        if split_node.target not in (
            aten.split.Tensor,
            aten.split_with_sizes.default,
        ):
            return None
        # Decode scatter_dim — for split.Tensor: args = (input, size, dim?).
        # For split_with_sizes: args = (input, sizes, dim?).
        if not split_node.args or not isinstance(split_node.args[0], torch.fx.Node):
            return None
        split_input = split_node.args[0]
        if len(split_node.args) >= 3:
            scatter_dim = int(split_node.args[2])
        else:
            scatter_dim = int(split_node.kwargs.get("dim", 0))
        return split_input, scatter_dim

    rs_target = c10d.reduce_scatter_tensor.default
    # Each candidate: (rs_node, walkback_start, producer_mm, scatter_dim_in_mm_output).
    candidates: list[tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, int]] = []
    total_rs = 0
    for node in gm.graph.nodes:
        if node.target is not rs_target:
            continue
        total_rs += 1
        rs_input = node.args[0]
        if not isinstance(rs_input, torch.fx.Node):
            continue
        # If the RS input is a cat that comes from a split (scatter_dim>0
        # decomposition), peel the split->cat off and continue walking
        # back from the split's input. Record the scatter_dim implied
        # by the split.
        walk_start: torch.fx.Node = rs_input
        forced_scatter_dim: int | None = None
        decoded = _decode_split_cat(rs_input)
        if decoded is not None:
            walk_start, forced_scatter_dim = decoded
        producer_mm = _trace_back_to_mm(walk_start)
        if producer_mm is None:
            continue
        # Conservative: only fuse if mm has a single user along this chain,
        # since the fused op doesn't return the mm result.
        if len(producer_mm.users) != 1:
            continue
        candidates.append(
            (node, walk_start, producer_mm, forced_scatter_dim or 0)
        )

    if not candidates:
        return gm

    # Register symmetric memory for any group we are about to fuse on.
    groups: set[str] = set()
    for rs_node, _, _, _ in candidates:
        groups.add(get_group_name(rs_node))
    for pg in groups:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            enable_symm_mem_for_group(pg)

    def _prod(xs) -> int:
        out = 1
        for x in xs:
            out *= int(x)
        return out

    fused = 0
    skipped_no_wait = 0
    skipped_shape = 0
    for rs_node, rs_input, producer_mm, forced_scatter_dim in candidates:
        # rs_node signature: reduce_scatter_tensor(input, reduce_op, group_size, group_name)
        if len(rs_node.args) < 4:
            continue
        _, reduce_op, _, group_name = rs_node.args[:4]
        # Find the wait_tensor user of the RS.
        wait_user = None
        for user in rs_node.users:
            if user.target is c10d.wait_tensor.default:
                wait_user = user
                break
        if wait_user is None:
            skipped_no_wait += 1
            continue

        # Infer scatter_dim in the mm-output space. The fused op operates
        # on the mm output (2D: M x N). If the chain transposes the mm
        # output, scatter_dim in mm-output space is flipped.
        mm_val = producer_mm.meta.get("val")
        rs_input_val = rs_input.meta.get("val")
        if mm_val is None or rs_input_val is None:
            skipped_shape += 1
            continue
        mm_shape = tuple(mm_val.shape)
        rs_in_shape = tuple(rs_input_val.shape)
        # We expect mm_shape to be 2D (M, N).
        if len(mm_shape) != 2:
            skipped_shape += 1
            continue
        # rs_input (the split-input or pre-RS tensor) may be reshaped or
        # transposed from mm output. Determine scatter_dim in mm-output
        # space.
        if forced_scatter_dim:
            # The RS used a split->cat decomposition along a specific dim
            # in rs_input space. Map that dim back to mm-output space via
            # shape comparison (only handle trivial reshape that preserves
            # the scattered dim's identity).
            if (
                len(rs_in_shape) >= 1
                and rs_in_shape[-1] == mm_shape[1]
                and _prod(rs_in_shape[:-1]) == mm_shape[0]
            ):
                # Reshape leading dims -> M; last dim is N.
                scatter_dim = (
                    1 if forced_scatter_dim == len(rs_in_shape) - 1 else 0
                )
            elif rs_in_shape == mm_shape:
                scatter_dim = forced_scatter_dim
            else:
                skipped_shape += 1
                continue
        else:
            # No split->cat decomposition: rs_input either matches mm or
            # is a pure reshape/transpose of it.
            if rs_in_shape == mm_shape:
                scatter_dim = 0
            elif rs_in_shape == (mm_shape[1], mm_shape[0]):
                scatter_dim = 1
            elif (
                len(rs_in_shape) >= 1
                and rs_in_shape[-1] == mm_shape[1]
                and _prod(rs_in_shape[:-1]) == mm_shape[0]
            ):
                # Reshape that preserves N on last dim, collapses leading to M.
                scatter_dim = 0
            else:
                skipped_shape += 1
                continue

        # The fused op outputs in mm's A dtype and performs the reduce
        # in that dtype. If the original chain upcast (e.g. bf16 -> fp32
        # via _to_copy) before RS, the original RS reduced in the
        # upcast dtype for numerical stability (this is the TP
        # weight-grad pattern). Fusing would change the reduction
        # precision, so we skip those chains.
        mm_a_dtype = mm_val.dtype
        wait_val = wait_user.meta.get("val")
        downstream_dtype = wait_val.dtype if wait_val is not None else mm_a_dtype
        if downstream_dtype != mm_a_dtype:
            skipped_shape += 1
            continue

        # Compute the actual 2D shape of the fused op's runtime output.
        # The op returns ``A @ B`` reduce-scattered along ``scatter_dim``
        # in the (2D) mm-output space; runtime shape is
        # ``out_shape = [*A.shape[:-1], B.shape[1]]`` with
        # ``out_shape[scatter_dim] //= group.size()`` (see
        # ``_fused_matmul_reduce_scatter_impl`` in torch.distributed._symmetric_memory).
        # We stamp this 2D shape on the fused-op node and insert an
        # explicit reshape back to the original (3D) wait_tensor shape
        # so downstream consumers (and any regional-Inductor scoop) see
        # the correct shape on every node.
        group_size = int(rs_node.args[2])
        if scatter_dim == 0:
            fused_2d_shape = (mm_shape[0] // group_size, mm_shape[1])
        else:
            fused_2d_shape = (mm_shape[0], mm_shape[1] // group_size)
        # Create a FakeTensor for the fused op output by viewing wait_val
        # (same element count and dtype/device) into the actual 2D shape.
        fused_val = wait_val.view(*fused_2d_shape)

        with gm.graph.inserting_before(rs_node):
            fused_out = gm.graph.call_function(
                symm_mem.fused_matmul_reduce_scatter.default,
                args=(
                    producer_mm.args[0],
                    producer_mm.args[1],
                    reduce_op,
                    scatter_dim,
                    group_name,
                ),
            )
            # Fused-op node sees the *actual* 2D runtime shape.
            fused_out.meta["val"] = fused_val
            # Reshape back to the original (possibly 3D) wait_tensor
            # shape so downstream users see the same shape as before
            # fusion. Without this, downstream regional-Inductor scoops
            # that codegen ``assert_size_stride`` from meta crash with
            # a shape mismatch (e.g. ``wrong number of dimensions``).
            reshape_out = gm.graph.call_function(
                aten.reshape.default,
                args=(fused_out, list(wait_val.shape)),
            )
            reshape_out.meta["val"] = wait_val

        # Redirect users of the wait to the reshape (3D) view. The
        # fused op already returns the reduce-scattered result (waits
        # internally); the reshape just restores the original rank.
        wait_user.replace_all_uses_with(reshape_out)
        gm.graph.erase_node(wait_user)

        # If this was a split->cat path, erase the cat + getitems + split
        # nodes between the original rs_input and the walk_start (split's
        # input). After this block, the original RS input's args[0] is no
        # longer reachable through this RS.
        original_rs_input = rs_node.args[0]
        gm.graph.erase_node(rs_node)
        if forced_scatter_dim and original_rs_input is not rs_input:
            # original_rs_input is the cat node; its args[0] is list of getitems;
            # each getitem points at the split node; split node points at rs_input.
            cat_node = original_rs_input
            if isinstance(cat_node, torch.fx.Node) and len(cat_node.users) == 0:
                getitem_list = cat_node.args[0] if cat_node.args else []
                gm.graph.erase_node(cat_node)
                # Find the split node from any getitem.
                split_node = None
                if isinstance(getitem_list, (list, tuple)) and getitem_list:
                    first = getitem_list[0]
                    if isinstance(first, torch.fx.Node) and first.args:
                        candidate = first.args[0]
                        if isinstance(candidate, torch.fx.Node):
                            split_node = candidate
                for gi in getitem_list:
                    if isinstance(gi, torch.fx.Node) and len(gi.users) == 0:
                        gm.graph.erase_node(gi)
                if split_node is not None and len(split_node.users) == 0:
                    gm.graph.erase_node(split_node)

        # Walk back through the rs_input chain and erase nodes that no
        # longer have users. Stop at the mm.
        cur: torch.fx.Node | None = rs_input
        while isinstance(cur, torch.fx.Node) and cur is not producer_mm:
            nxt = cur.args[0] if cur.args else None
            if len(cur.users) == 0:
                gm.graph.erase_node(cur)
            cur = nxt if isinstance(nxt, torch.fx.Node) else None
        if len(producer_mm.users) == 0:
            gm.graph.erase_node(producer_mm)
        fused += 1

    if fused > 0 or skipped_shape > 0 or skipped_no_wait > 0:
        logger.info(
            f"Custom-fused {fused} mm->RS pairs into fused_matmul_reduce_scatter "
            f"(scanned={total_rs}, candidates={len(candidates)}, "
            f"skipped: no_wait={skipped_no_wait}, shape_or_dtype={skipped_shape})"
        )
    gm.graph.lint()
    gm.recompile()
    return gm


def annotate_swiglu_chains_for_regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Tag bf16 pointwise chains around silu/silu_backward for Inductor fusion.

    Seed with ``aten.silu``/``aten.silu_backward``, then iteratively expand to
    adjacent ``aten.mul.Tensor`` neighbors whose tensor inputs are all bf16.
    The connected component becomes a single Inductor region. Bitwise-safe
    because pure bf16 pointwise mul is elementwise — same FP order as the
    separate eager kernels.

    The backward SwiGLU template is a chain ``silu_backward -> mul -> mul``
    (and analogous forward ``silu -> mul``); 32 Llama3 layers gives ~3-4 ops
    per layer that benefit from a single fused Triton kernel.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    seed_targets = {aten.silu.default, aten.silu_backward.default}
    expand_targets = {aten.mul.Tensor}

    def _is_bf16_node(n: torch.fx.Node) -> bool:
        v = n.meta.get("val")
        if not isinstance(v, FakeTensor):
            return False
        return v.dtype == torch.bfloat16

    def _all_bf16_inputs(n: torch.fx.Node) -> bool:
        for a in n.args:
            if isinstance(a, torch.fx.Node):
                v = a.meta.get("val")
                if isinstance(v, FakeTensor) and v.dtype != torch.bfloat16:
                    return False
        return True

    tagged: set[torch.fx.Node] = set()
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in seed_targets and _is_bf16_node(node):
            tagged.add(node)

    # Iteratively expand to bf16 muls adjacent to tagged ops until fixed point.
    changed = True
    while changed:
        changed = False
        for node in gm.graph.nodes:
            if node in tagged:
                continue
            if node.op != "call_function":
                continue
            if node.target not in expand_targets:
                continue
            if not _is_bf16_node(node) or not _all_bf16_inputs(node):
                continue
            neighbors_tagged = any(
                isinstance(a, torch.fx.Node) and a in tagged for a in node.args
            ) or any(u in tagged for u in node.users)
            if neighbors_tagged:
                tagged.add(node)
                changed = True

    for node in tagged:
        node.meta.setdefault("custom", {})["compile_with_inductor"] = {}
    if tagged:
        logger.info(
            f"Tagged {len(tagged)} SwiGLU-chain pointwise nodes for regional Inductor compilation"
        )
    return gm


def annotate_residual_add_for_regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Tag bf16 ``aten.add.Tensor`` (residual adds) for regional_inductor.

    The Llama3 graph has many residual-add ops (post-attention and post-MLP)
    on bf16 activations.  These are pure pointwise bf16 kernels; tagging them
    for regional Inductor compilation lets Inductor fuse them into adjacent
    elementwise ops, recovering a few ms of kernel time at no numerical cost
    (bf16 add is commutative and associative-equivalent for the small chains
    that result).

    Only tags ``aten.add.Tensor`` nodes where both inputs are bf16 FakeTensors
    so we never touch fp32 master-grad accumulation or scalar adds.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    num_tagged = 0
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is not aten.add.Tensor:
            continue
        if len(node.args) < 2:
            continue
        a, b = node.args[0], node.args[1]
        if not (isinstance(a, torch.fx.Node) and isinstance(b, torch.fx.Node)):
            continue
        a_val = a.meta.get("val")
        b_val = b.meta.get("val")
        if not (isinstance(a_val, FakeTensor) and isinstance(b_val, FakeTensor)):
            continue
        if a_val.dtype != torch.bfloat16 or b_val.dtype != torch.bfloat16:
            continue
        node.meta.setdefault("custom", {})["compile_with_inductor"] = {}
        num_tagged += 1
    if num_tagged > 0:
        logger.info(
            f"Tagged {num_tagged} bf16 add.Tensor nodes for regional Inductor compilation"
        )
    return gm


def async_tensor_parallel_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
    """Pipeline TP collectives with matmuls via symmetric memory.

    Fuses all-gather + matmul into ``symm_mem.fused_all_gather_matmul``
    and matmul + reduce-scatter into
    ``symm_mem.fused_matmul_reduce_scatter``.
    """
    from torch._inductor.fx_passes.micro_pipeline_tp import micro_pipeline_tp_pass
    from torch._inductor.fx_passes.overlap_scheduling import get_group_name
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    # Ensure symmetric memory is registered for every collective PG in
    # the graph.  The upstream API is deprecated but the auto-registration
    # it promises has not landed yet, so the explicit call is still needed.
    collective_targets = {
        c10d.all_gather_into_tensor.default,
        c10d.reduce_scatter_tensor.default,
    }
    registered: set[str] = set()
    for node in gm.graph.nodes:
        if node.target not in collective_targets:
            continue
        pg = get_group_name(node)
        if pg not in registered:
            registered.add(pg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                enable_symm_mem_for_group(pg)

    micro_pipeline_tp_pass(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def compile_time_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
    *,
    use_cudagraph: bool = False,
) -> list[Callable]:
    """Cleanup, FlexAttention annotation, and regional_inductor passes.

    If precompile is enabled, these are applied before serialization so
    that compiled Triton kernels are baked into the artifact. Otherwise
    they run at trace time via ``construct_default_graph_passes``.

    cudagraph is excluded because it needs to re-capture the graph into
    an in-memory CUDA graph at runtime.

    ``auto_overlap_bucketing_pass`` is preceded by ``overlap_fsdp_ag_rs_pass``
    (gated by ``compile.enable_fsdp_ag_rs_overlap``) so that forward+backward
    all-gathers end up on a separate CUDA stream from reduce-scatters
    (enabling AG/RS overlap in backward) before the auto-bucketer reorders
    the schedule.
    """
    from torchtitan.models.common.attention import FlexAttention

    passes: list[Callable] = [
        remove_detach_pass,
        remove_identity_view_pass,
        remove_identity_slice_pass,
        normalize_view_ops_as_reshape,
        remove_redundant_contiguous_clone_pass,
        # Custom mm->RS fusion BEFORE bucketing so the matcher sees the
        # original mm->{reshape|_to_copy|...}->RS chains intact.
        fuse_mm_reduce_scatter_pass,
    ]
    # Preserve AG/RS overlap (separate streams for AG vs RS) before auto-bucketing.
    if config.compile.enable_fsdp_ag_rs_overlap:
        passes.append(overlap_fsdp_ag_rs_pass)
    passes.append(auto_overlap_bucketing_pass)
    # Auto-bucketing's pre-bucketing rewrite may introduce new redundant
    # clone(contiguous_format) calls along the unpack chain
    # (_pre_bucket_all_gather -> wait_tensor -> ... -> unpack -> mm).
    # Run cleanup a second time to catch any that appear post-bucketing.
    passes.append(remove_redundant_contiguous_clone_pass)
    if config.parallelism.enable_async_tensor_parallel:
        passes.append(async_tensor_parallel_pass)

    inductor_compilation = config.compile.inductor_compilation
    if inductor_compilation == "full":
        # Compile the entire graph into optimized Triton kernels. Must
        # be terminal — the FX graph is no longer authoritative after
        # this pass, so custom_codegen_pass and
        # insert_kernel_annotations_pass cannot follow.
        passes.append(full_inductor_compilation_pass)
    if inductor_compilation == "regional":
        # FlexAttention HOPs must be compiled (via regional_inductor) to
        # produce bitwise identical results to the eager Trainer path.
        # When left uncompiled, flex_attention still runs correctly but
        # produces different numerical results.
        passes.append(
            functools.partial(
                annotate_flex_attention_for_regional_inductor_pass,
                flex_compile_config=FlexAttention.inductor_configs,
            )
        )
        # Tag bf16 SwiGLU chains (silu/silu_backward + adjacent bf16 muls)
        # before residual-add tagging so adds that are NOT part of the
        # SwiGLU chain still get picked up by the residual-add pass.
        passes.append(annotate_swiglu_chains_for_regional_inductor_pass)
        # Tag bf16 residual adds for regional Inductor (numerics-preserving:
        # pure pointwise bf16 fusion). Must run after the flex annotation
        # pass and before ``regional_inductor_pass`` so the tagger sees
        # the final graph but the compiler picks up the new tags.
        passes.append(annotate_residual_add_for_regional_inductor_pass)
        # Performance passes that may change numerics.
        if config.compile.numerics_changing_optim:
            from torchtitan.experiments.graph_trainer.performance_passes import (
                annotate_rmsnorm_for_regional_inductor_pass,
            )

            passes.append(annotate_rmsnorm_for_regional_inductor_pass)
        passes.append(regional_inductor_pass)
        if use_cudagraph:
            # Must run before custom_codegen_pass (last in pre_passes)
            # which replaces the GraphModule's forward().
            # Also must run before cudagraph_pass.
            passes.append(insert_kernel_annotations_pass)
        # TODO: Switch to upstream PyTorch implementation when
        # https://github.com/pytorch/pytorch/pull/178246 lands.
        # custom_codegen_pass saves the FX graph to disk for:
        # 1. Debugging: inspect the generated graph code directly
        # 2. Profiling provenance: dual-path codegen with _RecordFunctionFast
        #    gives fine-grained operator-level attribution in profiler traces
        # 3. User-editable codegen: users can directly modify the generated
        #    program on disk for fine-grain scheduling optimizations, with
        #    hot-reload picking up changes at runtime
        passes.append(custom_codegen_pass)
    return passes


def construct_default_graph_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
) -> list[Callable]:
    """Build the pass list for the aot_fx_trace path.

    When ``precompile_artifact_dir`` is unset, returns the full list: cleanup,
    FlexAttention annotation, regional_inductor, and cudagraph.

    When ``precompile_artifact_dir`` is set, the artifact has graph
    transformed during precompile phase, so only cudagraph is returned.
    """
    want_cudagraph = "cudagraph_pass" not in config.compile.disable_passes

    has_precompile_artifact = bool(config.compile.precompile_artifact_dir)

    passes: list[Callable] = []
    if not has_precompile_artifact:
        passes.extend(
            compile_time_passes(traced_result, config, use_cudagraph=want_cudagraph)
        )

    if want_cudagraph:
        static_input_indices = list(range(traced_result.num_static_inputs))
        passes.append(
            functools.partial(
                cudagraph_pass,
                is_forward=True,
                static_input_indices=static_input_indices,
                tensor_input_indices=traced_result.tensor_input_indices,
            )
        )
    return passes


def _get_pass_name(pass_fn: Callable) -> str:
    return (
        pass_fn.func.__name__
        if isinstance(pass_fn, functools.partial)
        else pass_fn.__name__
    )


def _filter_disabled_passes(
    passes: list[Callable], disable_names: list[str]
) -> list[Callable]:
    """Remove passes whose names exactly match any entry in ``disable_names``."""
    disable_set = set(disable_names)
    filtered = []
    skipped = []
    for pass_fn in passes:
        name = _get_pass_name(pass_fn)
        if name in disable_set:
            skipped.append(name)
        else:
            filtered.append(pass_fn)
    if skipped:
        logger.info(f"Disabled {len(skipped)} graph passes: {skipped}")
    return filtered


def apply_graph_passes(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    passes: list[Callable],
    *,
    compile_config: "GraphTrainerCompileConfig | None" = None,
) -> torch.fx.GraphModule:
    """Apply graph passes to the traced fwd+bwd graph.

    Args:
        gm: The traced forward+backward graph module.
        example_inputs: Example (fake) inputs matching the graph signature.
        passes: Ordered list of pass callables, each with signature
            ``(gm, example_inputs, **kwargs) -> gm``.
        compile_config: Optional compile config. When provided and
            ``debug_graph_passes`` is True, logs timing, op-count diffs,
            and before/after graphs to tlparse for each pass.
    """
    debug = compile_config is not None and compile_config.debug_graph_passes
    disable_patterns = (
        compile_config.disable_passes if compile_config is not None else []
    )
    if disable_patterns:
        passes = _filter_disabled_passes(passes, disable_patterns)
    pass_names = [_get_pass_name(pass_fn) for pass_fn in passes]
    pass_list = "\n  ".join(f"{i}. {name}" for i, name in enumerate(pass_names, 1))
    logger.info(f"Applying {len(passes)} graph passes:\n  {pass_list}")
    all_passes_start = time.perf_counter()
    tlparse_log_graph_pass(gm, graph_name="make_fx_graph_traced", debug=debug)
    for pass_fn in passes:
        pass_name = _get_pass_name(pass_fn)
        if debug:
            tlparse_log_graph_pass(gm, graph_name=f"before_{pass_name}", debug=debug)
            before_snapshot = snapshot_graph(gm)
            start = time.perf_counter()
        gm = pass_fn(gm, example_inputs)
        assert isinstance(
            gm, torch.fx.GraphModule
        ), f"Pass {pass_name} returned {type(gm).__name__}, expected GraphModule"
        if debug:
            elapsed = time.perf_counter() - start
            logger.info(f"Pass {pass_name} took {elapsed:.3f}s")
            tlparse_log_graph_pass(gm, graph_name=f"after_{pass_name}", debug=debug)
            after_snapshot = snapshot_graph(gm)
            log_graph_diff(before_snapshot, after_snapshot, pass_name)
    all_passes_elapsed = time.perf_counter() - all_passes_start
    logger.info(f"All {len(passes)} graph passes took {all_passes_elapsed:.3f}s")
    return gm


def tlparse_log_graph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    graph_name: str,
    debug: bool = False,
) -> torch.fx.GraphModule:
    """Log the transformed graph to tlparse via trace_structured.

    This pass should be added as the last transform in fwd/bwd_transforms
    so that the logged graph reflects all prior transformations.

    Args:
        gm: The graph module to log.
        example_inputs: The example inputs (unused, required by protocol).
        graph_name: The name for this graph artifact
            (e.g. "aot_forward_graph_transformed").
        debug: When True, include additional metadata in the printed nodes.

    Returns:
        The graph module unchanged.
    """
    additional_meta = ["autograd_backward"]
    if debug:
        additional_meta.append("seq_nr")

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": graph_name,
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
            additional_meta=additional_meta,
        ),
        expect_trace_id=False,
    )

    return gm
