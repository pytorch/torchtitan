# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compiler passes for the compiler toolkit.

This module provides various compiler passes that can be applied to graph modules
during compilation. Passes can be selected and configured via job config.

Pass Types:
- Joint custom passes: Applied to the joint forward-backward graph before partitioning
- Compiler passes: Applied to the partitioned forward/backward graphs
"""
import operator
from collections.abc import Sequence
from typing import Any

import torch
from torch._functorch.aot_autograd import JointWithDescriptors
from torch._guards import TracingContext
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)
from torch._inductor.fx_passes.overlap_manual_scheduling import manual_overlap_bucketing
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing
from torch.fx.passes.regional_inductor import regional_inductor
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.experiments.compiler_toolkit.cudagraph import (
    CUDAGraphWrapper,
    get_static_input_indices,
)
from torchtitan.experiments.simple_fsdp.reshard_after_forward import (
    annotate_fsdp_all_gather,
)
from torchtitan.tools.logging import logger


def autobucketing_reordering_pass(
    gm: torch.fx.GraphModule, example_inputs=None
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
    gm: torch.fx.GraphModule, example_inputs, fsdp_manual_buckets
) -> torch.fx.GraphModule:
    """
    Apply aten-level manual bucketing and reordering optimization.
    """
    manual_overlap_bucketing(
        gm, module_bucket_plans=fsdp_manual_buckets, insert_overlap_deps=False
    )
    gm.recompile()
    return gm


def regional_inductor_pass(
    gm: torch.fx.GraphModule, example_inputs
) -> torch.fx.GraphModule:
    """
    Apply regional inductor compilation based on user annotation.
    """
    return regional_inductor(gm, example_inputs)


def cudagraph_pass(
    gm: torch.fx.GraphModule, example_inputs: Sequence[Any], is_forward: bool
) -> torch.fx.GraphModule:
    """
    Apply cudagraph.

    This pass wraps the forward function with cudagraph during compilation and does
    not record cudagraph until runtime.
    - For the first run, it will warm up operators such as nccl.
    - For the second run, it will record cudagraph and replay cudagraph.
    - For the following runs, it will replay cudagraph.
    """
    static_input_indices = get_static_input_indices(gm, is_forward)
    gm.forward = CUDAGraphWrapper(gm.forward, example_inputs, static_input_indices)
    return gm


def validate_flex_attn_annotation_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Verify user annotations show up in the graph."""
    for node in gm.graph.nodes:
        if node.target in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            assert "compile_with_inductor" in node.meta.get("custom", {})
    return gm


# Default set of ops whose outputs should be saved (not recomputed) during
# activation checkpointing. These are compute-intensive or communication ops
# where recomputation is expensive.
DEFAULT_SAC_SAVE_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch.ops.torch_attn._varlen_attn,
    torch._higher_order_ops.inductor_compiled_code,
}


def apply_sac_pass(
    gm: torch.fx.GraphModule,
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
            Defaults to DEFAULT_SAC_SAVE_OPS if None.

    Returns:
        The annotated graph module
    """
    if op_list_to_save is None:
        op_list_to_save = DEFAULT_SAC_SAVE_OPS

    nodes = list(gm.graph.nodes)
    output_node = nodes[-1].all_input_nodes[0]
    mm_count = 0

    for node in nodes:
        if node.op != "call_function" or node.target is operator.getitem:
            continue

        node.meta["ac_graph_id"] = 0

        if node.target is torch.ops.aten.mm.default:
            mm_count += 1
            # Save every odd mm, recompute every even mm
            if mm_count % 2 == 0:
                node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
            else:
                node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        elif node.target in op_list_to_save:
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        else:
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE

        if node is output_node:
            break

    gm.recompile()
    logger.info(
        "Applied selective activation checkpointing (SAC) graph pass "
        f"({mm_count} mm ops found, {mm_count - mm_count // 2} saved)"
    )
    return gm


# Apply activation checkpointing on joint graph before partitioner
def fsdp_reshard_after_fwd_pass(
    gm: torch.fx.GraphModule, reshard_after_forward: bool
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
    model: torch.nn.Module,
    joint_with_descriptors: JointWithDescriptors,
    forward_inputs: tuple,
    tracing_context: TracingContext,
) -> torch.fx.GraphModule:
    """
    Apply Inductor decompositions to the joint graph.

    This pass applies decompositions to the joint forward-backward graph using make_fx.
    It unwraps tensor subclasses (like DTensor) and retraces the graph with decompositions
    applied, while preserving metadata required by the partitioner.

    Args:
        gm: The joint graph module
        model: The parallelized model
        joint_with_descriptors: The joint graph with descriptors
        forward_inputs: Forward input arguments (may be DTensors)
        tracing_context: The tracing context from original joint graph capture

    Returns:
        The joint graph with decompositions applied
    """
    from torch._functorch._aot_autograd.descriptors import DummyAOTInput
    from torch._functorch._aot_autograd.subclass_utils import unwrap_tensor_subclasses
    from torch._inductor.decomposition import select_decomp_table
    from torch.fx.experimental.proxy_tensor import make_fx

    logger.info("Applying decompositions to joint graph")

    decomp_table = select_decomp_table()

    # Get traced tangents metadata
    traced_tangents = joint_with_descriptors._aot_state.fw_metadata.traced_tangents

    # Collect all inputs: params, buffers, forward inputs, tangents
    param_inputs = list(model.parameters())
    buffer_inputs = list(model.buffers())
    primals = param_inputs + buffer_inputs + list(forward_inputs)
    tangents = list(traced_tangents)

    # Create dummy descriptors for unwrapping
    primals_descs = [DummyAOTInput(i) for i in range(len(primals))]
    tangents_descs = [DummyAOTInput(i + len(primals)) for i in range(len(tangents))]

    # Unwrap tensor subclasses (DTensor -> _local_tensor)
    primals_unwrapped, _ = unwrap_tensor_subclasses(
        primals, primals_descs, append_symints=False
    )
    tangents_unwrapped, _ = unwrap_tensor_subclasses(
        tangents, tangents_descs, append_symints=False
    )

    # Verify unwrapped tensor shapes match joint graph placeholders
    all_inputs = primals_unwrapped + tangents_unwrapped
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

    if len(all_inputs) != len(placeholders):
        raise RuntimeError(
            f"Input count mismatch: {len(all_inputs)} inputs vs {len(placeholders)} placeholders"
        )

    shape_mismatches = []
    for i, (inp, ph) in enumerate(zip(all_inputs, placeholders)):
        if hasattr(inp, "shape") and "val" in ph.meta:
            expected_shape = ph.meta["val"].shape
            actual_shape = inp.shape
            if expected_shape != actual_shape:
                shape_mismatches.append(
                    f"  {ph.target}: expected {expected_shape}, got {actual_shape}"
                )

    if shape_mismatches:
        logger.error(f"Shape mismatches found ({len(shape_mismatches)}):")
        for msg in shape_mismatches:
            logger.error(msg)
        raise RuntimeError(
            "Unwrapped tensor shapes don't match joint graph placeholders."
        )

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

        fake_mode = detect_fake_mode(primals_unwrapped)

    # Use make_fx with the original fake mode to retrace with decompositions
    with fake_mode:
        decomposed_gm = make_fx(
            gm,
            decomposition_table=decomp_table,
            _allow_non_fake_inputs=False,
        )(primals_unwrapped, tangents_unwrapped)

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
    gm: torch.fx.GraphModule, example_inputs
) -> torch.fx.GraphModule:
    """
    Apply full Inductor compilation with code generation.

    This pass uses compile_fx_inner to generate optimized code for the graph.

    Args:
        gm: The graph module (forward or backward)
        example_inputs: Example inputs for compilation

    Returns:
        The compiled graph module
    """
    return compile_fx_inner(gm, example_inputs)


def reassign_to_pg_pass(
    gm: torch.fx.GraphModule,
    example_inputs,
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


# Registry mapping pass names to pass functions
AVAILABLE_COMPILER_PASSES = {
    "autobucketing_reordering": autobucketing_reordering_pass,
    "transformer_block_bucketing": transformer_block_bucketing_reordering_pass,
    "regional_inductor": regional_inductor_pass,
    "cudagraph": cudagraph_pass,
    "full_inductor_compilation": full_inductor_compilation_pass,
}

# Registry for joint custom passes (applied before partitioning)
AVAILABLE_JOINT_PASSES = {
    "inductor_decomposition": inductor_decomposition_pass,
    "fsdp_reshard_after_fwd": fsdp_reshard_after_fwd_pass,
    "validate_flex_attn_annotation": validate_flex_attn_annotation_pass,
    "apply_sac": apply_sac_pass,
}
