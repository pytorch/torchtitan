# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compiler passes for graph_trainer training.

This module provides various compiler passes that can be applied to graph modules
during compilation. Passes can be selected and configured via job config.

Pass Types:
- Joint custom passes: Applied to the joint forward-backward graph before partitioning
- Compiler passes: Applied to the partitioned forward/backward graphs
"""

import operator
from collections import defaultdict

import torch
from torch._functorch.aot_autograd import JointWithDescriptors
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)
from torch._inductor.fx_passes.overlap_manual_scheduling import manual_overlap_bucketing
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing
from torch._inductor.output_code import OutputCode
from torch._logging import trace_structured
from torch.fx.passes.regional_inductor import regional_inductor
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.activation_checkpoint import _get_save_ops
from torchtitan.experiments.graph_trainer.common_utils import _AC_REGION_ID
from torchtitan.experiments.graph_trainer.reshard_after_forward import (
    annotate_fsdp_all_gather,
)
from torchtitan.tools.logging import logger


def apply_default_graph_passes(
    gm: torch.fx.GraphModule, example_inputs: tuple
) -> torch.fx.GraphModule:
    """Entry point for optimizing the traced fwd+bwd graph.

    Called by GraphTrainer after tracing to apply graph-level optimization
    passes before execution. Individual passes are defined below.
    """
    gm = tlparse_log_graph_pass(gm, example_inputs, graph_name="make_fx_graph_traced")

    gm = remove_detach_pass(gm, example_inputs)
    gm = remove_identity_view_pass(gm, example_inputs)
    gm = remove_identity_slice_pass(gm, example_inputs)
    gm = autobucketing_reordering_pass(gm, example_inputs)

    return gm


def remove_detach_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove identity-like detach nodes from the graph.

    In a traced graph, `aten.detach.default` is semantically a no-op (there is
    no autograd context). Replacing each detach node with its input reduces
    graph size and may improve scheduling.
    """
    count = 0
    for node in list(gm.graph.nodes):
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.detach.default
        ):
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
            count += 1
    if count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"Removed {count} detach nodes")
    return gm


def remove_identity_view_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove identity _unsafe_view ops where output shape equals input shape.

    In the traced graph, _unsafe_view(tensor, shape) where shape matches the
    input tensor's shape is a no-op. Removing these reduces graph size.
    """
    VIEW_OPS = {
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    }
    count = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in VIEW_OPS:
            continue
        # Check if the output shape matches the input shape via tensor metadata
        input_node = node.args[0]
        if not isinstance(input_node, torch.fx.Node):
            continue
        input_val = input_node.meta.get("val")
        output_val = node.meta.get("val")
        if input_val is None or output_val is None:
            continue
        if (
            hasattr(input_val, "shape")
            and hasattr(output_val, "shape")
            and input_val.shape == output_val.shape
        ):
            node.replace_all_uses_with(input_node)
            gm.graph.erase_node(node)
            count += 1
    if count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"Removed {count} identity view/reshape nodes")
    return gm


def remove_identity_slice_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove identity slice ops that select the full dimension.

    `aten.slice.Tensor(input, dim, 0, end)` where end >= input.size(dim)
    is a no-op — it returns the full tensor along that dimension.
    """
    count = 0
    for node in list(gm.graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.slice.Tensor
        ):
            continue
        input_node = node.args[0]
        if not isinstance(input_node, torch.fx.Node):
            continue
        input_val = input_node.meta.get("val")
        if input_val is None or not hasattr(input_val, "shape"):
            continue
        # slice.Tensor(input, dim=0, start=0, end=9223372036854775807, step=1)
        dim = node.args[1] if len(node.args) > 1 else 0
        start = node.args[2] if len(node.args) > 2 else 0
        end = node.args[3] if len(node.args) > 3 else input_val.shape[dim]
        step = node.args[4] if len(node.args) > 4 else 1
        if start == 0 and step == 1 and end >= input_val.shape[dim]:
            node.replace_all_uses_with(input_node)
            gm.graph.erase_node(node)
            count += 1
    if count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"Removed {count} identity slice nodes")
    return gm


def autobucketing_reordering_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple | None = None
) -> torch.fx.GraphModule:
    """
    Apply autobucketing and reordering optimization.

    This pass applies schedule_overlap_bucketing with collective_bucketing enabled
    to optimize comm/compute overlap patterns in the graph.
    """
    schedule_overlap_bucketing(gm, collective_bucketing=True)
    gm.recompile()
    return gm


def transformer_block_bucketing_reordering_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    fsdp_manual_buckets,
) -> torch.fx.GraphModule:
    """
    Apply aten-level manual bucketing and reordering optimization.
    """
    manual_overlap_bucketing(
        gm, module_bucket_plans=fsdp_manual_buckets, insert_overlap_deps=False
    )
    gm.recompile()
    return gm


def _ops_filter_with_distributed(name: str) -> bool:
    """Ops filter that allows distributed collective ops for serialization.

    The default GraphPickler ops filter only allows aten and fbgemm ops.
    SimpleFSDP uses _c10d_functional collectives that must also be
    allowed for the graph to serialize correctly.
    """
    return name.startswith(
        (
            "torch.ops.aten",
            "torch.ops.fbgemm",
            "torch.ops._c10d_functional",
        )
    )


def regional_inductor_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple, *, serializable: bool = False
) -> torch.fx.GraphModule:
    """
    Apply regional inductor compilation based on user annotation.

    When serializable=True (precompile mode), sets force_autograd_cache
    so that regional_inductor wraps its output in RegionalOutputCode,
    and overrides the ops filter to allow distributed collective ops.
    """
    if serializable:
        with torch._functorch.config.patch("force_autograd_cache", True):
            result = regional_inductor(gm, example_inputs)
        from torch._inductor.output_code import RegionalOutputCode

        # Override the ops filter after compilation so that
        # serialization (which happens later) allows distributed
        # collective ops like _c10d_functional through GraphPickler.
        if isinstance(result, RegionalOutputCode):
            result._ops_filter = _ops_filter_with_distributed
        else:
            logger.warning(
                "regional_inductor with serializable=True did not produce "
                "RegionalOutputCode; distributed ops may not serialize correctly."
            )
        return result
    return regional_inductor(gm, example_inputs)


def cudagraph_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple, *, is_forward: bool
) -> torch.fx.GraphModule:
    """
    Apply cudagraph.

    This pass wraps the forward function with cudagraph during compilation and does
    not record cudagraph until runtime.
    - For the first run, it will warm up operators such as nccl.
    - For the second run, it will record cudagraph and replay cudagraph.
    - For the following runs, it will replay cudagraph.
    """
    # Lazy import: cudagraph.py runs init_global_graph_pool() at import time,
    # which must happen after torch.cuda.set_device(local_rank).
    from torchtitan.experiments.graph_trainer.cudagraph import (
        CUDAGraphWrapper,
        get_static_input_indices,
    )

    static_input_indices = get_static_input_indices(gm, is_forward)
    gm.forward = CUDAGraphWrapper(gm.forward, example_inputs, static_input_indices)
    return gm


def validate_flex_attn_annotation_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple | None = None
) -> torch.fx.GraphModule:
    """Verify user annotations show up in the graph."""
    for node in gm.graph.nodes:
        if node.target in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            assert "compile_with_inductor" in node.meta.get("custom", {})
    return gm


def apply_sac_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    op_list_to_save: set | None = None,
) -> torch.fx.GraphModule:
    """
    Apply selective activation checkpointing on the joint graph.

    This pass iterates over all call_function nodes in the joint graph and annotates
    each with a CheckpointPolicy. Ops in ``op_list_to_save`` are marked MUST_SAVE
    (their outputs are kept as activations for the backward pass), while all other
    ops are marked PREFER_RECOMPUTE (their outputs may be discarded and recomputed
    during backward).

    To reduce memory further, every second ``mm`` op is marked PREFER_RECOMPUTE
    instead of MUST_SAVE, matching the behavior of the eager selective AC policy
    in ``torchtitan.distributed.activation_checkpoint``.

    The annotations are later consumed by the min-cut partitioner
    (``min_cut_rematerialization_partition``) to split the joint graph into separate
    forward and backward graphs.

    Usage: set ``--compile.joint_passes apply_sac``.

    Args:
        gm: The joint forward-backward graph module
        op_list_to_save: Set of op targets whose outputs should be saved.
            Defaults to ``torchtitan.distributed.activation_checkpoint._get_save_ops()``
            if None.

    Returns:
        The annotated graph module
    """
    if op_list_to_save is None:
        op_list_to_save = _get_save_ops()

    mm_count = 0
    ac_region_stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"save": 0, "recompute": 0}
    )

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        if node.target in (
            operator.getitem,
            torch.ops._c10d_functional.wait_tensor.default,
        ):
            # Propagate recompute tag from the parent node:
            # - getitem: When a node returns a tuple/list (e.g., rmsnorm, sdpa),
            #   it is followed by getitem nodes that extract individual elements.
            #   They inherit the parent's recompute tag, otherwise they will be
            #   exposed as graph outputs and saved for backwards unnecessarily.
            # - wait_tensor: Semantically tied to its parent async collective
            #   (e.g., reduce_scatter_tensor, all_gather_into_tensor) and must
            #   share the same save/recompute decision.
            parent = node.args[0]
            if isinstance(parent, torch.fx.Node) and "recompute" in parent.meta:
                node.meta["recompute"] = parent.meta["recompute"]
                node.meta["ac_graph_id"] = parent.meta.get("ac_graph_id", 0)
            continue

        custom_meta = node.meta.get("custom", {})
        ac_region_id = custom_meta.get(_AC_REGION_ID, 0)
        node.meta["ac_graph_id"] = ac_region_id

        if node.target is torch.ops.aten.mm.default:
            mm_count += 1
            # Save every odd mm, recompute every even mm
            if mm_count % 2 == 0:
                policy = CheckpointPolicy.PREFER_RECOMPUTE
            else:
                policy = CheckpointPolicy.MUST_SAVE
        elif node.target in op_list_to_save:
            policy = CheckpointPolicy.MUST_SAVE
        else:
            policy = CheckpointPolicy.PREFER_RECOMPUTE

        node.meta["recompute"] = policy
        if policy == CheckpointPolicy.MUST_SAVE:
            ac_region_stats[ac_region_id]["save"] += 1
        else:
            ac_region_stats[ac_region_id]["recompute"] += 1

    gm.recompile()
    logger.info("Applied selective activation checkpointing (SAC) graph pass.")
    for ac_region_id in sorted(ac_region_stats):
        stats = ac_region_stats[ac_region_id]
        logger.info(
            f"  AC region {ac_region_id}: "
            f"{stats['save']} nodes annotated with MUST_SAVE, "
            f"{stats['recompute']} nodes annotated with PREFER_RECOMPUTE"
        )
    return gm


# Apply activation checkpointing on joint graph before partitioner
def fsdp_reshard_after_fwd_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    reshard_after_forward: bool,
) -> torch.fx.GraphModule:
    # this pass implements simplefsdp's fsdp_reshard_after_forward behavior
    # when fsdp_reshard_after_forward set to True, it will annotate simple_fsdp AG
    #   to CheckpointPolicy.MUST_RECOMPUTE.
    # when fsdp_reshard_after_forward set to False, it will annotate simple_fsdp AG
    #   to CheckpointPolicy.MUST_SAVE.
    gm = annotate_fsdp_all_gather(gm, reshard_after_forward)
    gm.recompile()
    return gm


def inductor_decomposition_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    joint_with_descriptors: JointWithDescriptors,
) -> torch.fx.GraphModule:
    """
    Apply Inductor decompositions to the joint graph.

    This pass applies decompositions to the joint forward-backward graph using make_fx.
    It reads fake tensor inputs from placeholder metadata and retraces the graph with
    decompositions applied, while preserving metadata required by the partitioner.

    Args:
        gm: The joint graph module
        joint_with_descriptors: The joint graph with descriptors

    Returns:
        The joint graph with decompositions applied
    """
    from torch._inductor.decomposition import select_decomp_table
    from torch.fx.experimental.proxy_tensor import make_fx

    logger.info("Applying decompositions to joint graph")

    decomp_table = select_decomp_table()

    # Build fake inputs directly from the joint graph placeholders' metadata.
    # This handles all inputs including effect tokens (e.g. from MoE load
    # balancing copy_ mutations) that AOT Autograd prepends as placeholders,
    # as well as opaque inputs (e.g. DeviceMesh FakeScriptObjects) that the
    # graph lifts when compile-on-one-rank is enabled.
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    all_inputs = []
    for ph in placeholders:
        val = ph.meta.get("val")
        if val is None:
            raise RuntimeError(f"Placeholder {ph.target} has no 'val' metadata")
        all_inputs.append(val)

    # The joint graph forward() takes (primals, tangents) as two list args.
    # Use the graph's _in_spec (set by AOTAutograd during joint export) to
    # determine the correct split point rather than
    # fw_metadata.traced_tangents, because the latter only counts tensor
    # tangents and misses opaque inputs (e.g. DeviceMesh objects) that may
    # appear as additional placeholders when compile-on-one-rank is enabled.
    num_primals = gm._in_spec.child(0).num_children
    primals_fake = all_inputs[:num_primals]
    tangents_fake = all_inputs[num_primals:]

    # Get the FakeTensorMode from the original joint graph
    fake_mode = None
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "fake_mode"):
                fake_mode = val.fake_mode
                break

    if fake_mode is None:
        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode(all_inputs)

    # Use make_fx with the original fake mode to retrace with decompositions
    with fake_mode:
        decomposed_gm = make_fx(
            gm,
            decomposition_table=decomp_table,
            _allow_non_fake_inputs=False,
        )(primals_fake, tangents_fake)

    # Copy metadata from original placeholders to decomposed placeholders
    orig_placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    decomp_placeholders = [
        n for n in decomposed_gm.graph.nodes if n.op == "placeholder"
    ]

    if len(orig_placeholders) != len(decomp_placeholders):
        raise RuntimeError(
            f"Placeholder count mismatch: {len(orig_placeholders)} vs {len(decomp_placeholders)}"
        )

    for orig, decomp in zip(orig_placeholders, decomp_placeholders):
        # Copy all metadata from original to decomposed
        for key, value in orig.meta.items():
            if key not in decomp.meta:
                decomp.meta[key] = value

        # Rename decomposed placeholder to match original name
        decomp.target = orig.target
        decomp.name = orig.name

    decomposed_gm.recompile()
    logger.info("Decompositions applied successfully to joint graph")

    return decomposed_gm


def full_inductor_compilation_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple
) -> OutputCode:
    """
    Apply full Inductor compilation with code generation.

    This pass uses compile_fx_inner to generate optimized code for the graph.

    Args:
        gm: The graph module (forward or backward)
        example_inputs: Example inputs for compilation

    Returns:
        The compiled OutputCode from Inductor
    """
    # TODO: This pass returns OutputCode instead of GraphModule, violating the
    # unified graph pass signature convention. Should be addressed to comply.
    return compile_fx_inner(gm, example_inputs)


def reassign_to_pg_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    source_pg_name: str,
    target_pg_name: str,
) -> torch.fx.GraphModule:
    """
    Reassign all-gather nodes from one process group to another.

    This pass rewrites all-gather nodes whose PG matches ``source_pg_name`` to use
    ``target_pg_name`` instead.  Since each NCCL PG gets its own CUDA stream, this
    can be used to separate AG and RS onto different streams (e.g. for AG/RS
    overlap in the backward pass).

    Must be applied BEFORE bucketing passes so that bucketed all-gathers inherit
    the new PG name.

    Args:
        gm: The graph module (forward or backward)
        example_inputs: Example inputs (unused, required by pass interface)
        source_pg_name: The group_name of the process group to match
        target_pg_name: The group_name of the process group to assign
    """
    count = 0
    for node in gm.graph.nodes:
        if is_all_gather(node):
            # AG args: (input_tensor, group_size, group_name)
            if node.args[2] == source_pg_name:
                node.args = (node.args[0], node.args[1], target_pg_name)
                count += 1
    if count > 0:
        logger.info(
            f"Rewrote {count} all-gather node(s) from PG {source_pg_name} "
            f"to PG {target_pg_name}"
        )
    gm.recompile()
    return gm


def tlparse_log_graph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    graph_name: str,
) -> torch.fx.GraphModule:
    """Log the transformed graph to tlparse via trace_structured.

    This pass should be added as the last transform in fwd/bwd_transforms
    so that the logged graph reflects all prior transformations.

    Args:
        gm: The graph module to log.
        example_inputs: The example inputs (unused, required by protocol).
        graph_name: The name for this graph artifact
            (e.g. "aot_forward_graph_transformed").

    Returns:
        The graph module unchanged.
    """
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
        ),
        expect_trace_id=False,
    )

    return gm


# Registry mapping pass names to pass functions (for AOT mode fwd/bwd passes)
AVAILABLE_COMPILER_PASSES = {
    "auto_bucketing": autobucketing_reordering_pass,
    "transformer_block_bucketing": transformer_block_bucketing_reordering_pass,
    "regional_inductor": regional_inductor_pass,
    "cudagraph": cudagraph_pass,
    "full_inductor_compilation": full_inductor_compilation_pass,
}

# Registry for joint custom passes (applied before partitioning, AOT mode only)
AVAILABLE_JOINT_PASSES = {
    "inductor_decomposition": inductor_decomposition_pass,
    "fsdp_reshard_after_fwd": fsdp_reshard_after_fwd_pass,
    "validate_flex_attn_annotation": validate_flex_attn_annotation_pass,
    "apply_sac": apply_sac_pass,
}
