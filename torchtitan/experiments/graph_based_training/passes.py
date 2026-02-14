# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified compiler pass registry for graph-based training.

This module provides all compiler passes and a registry with metadata
(phase, supported modes) for each pass. Passes can be selected and
configured via job config.

Pass Types:
- Pre-partition passes: Applied to the joint forward-backward graph before partitioning
- Post-partition passes: Applied to the partitioned forward/backward graphs
"""

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch._functorch.aot_autograd import JointWithDescriptors
from torch._guards import TracingContext
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.fx_passes.overlap_manual_scheduling import manual_overlap_bucketing
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing
from torch.fx.passes.regional_inductor import regional_inductor

from torchtitan.tools.logging import logger

from .cudagraph import CUDAGraphWrapper, get_static_input_indices
from .reshard_after_forward import annotate_fsdp_all_gather


# ---------------------------------------------------------------------------
# Post-partition passes
# ---------------------------------------------------------------------------


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


def full_inductor_compilation_pass(
    gm: torch.fx.GraphModule, example_inputs
) -> torch.fx.GraphModule:
    """
    Apply full Inductor compilation with code generation.

    This pass uses compile_fx_inner to generate optimized code for the graph.
    """
    return compile_fx_inner(gm, example_inputs)


# ---------------------------------------------------------------------------
# Pre-partition (joint) passes
# ---------------------------------------------------------------------------


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


def fsdp_reshard_after_fwd_pass(
    gm: torch.fx.GraphModule, reshard_after_forward: bool
) -> torch.fx.GraphModule:
    """
    Annotate FSDP all-gather operations for reshard-after-forward behavior.

    When reshard_after_forward is True, annotates all-gather results as
    MUST_RECOMPUTE (freed after forward, recomputed in backward).
    When False, annotates as MUST_SAVE (kept in memory).
    """
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


# ---------------------------------------------------------------------------
# Pass registry
# ---------------------------------------------------------------------------


@dataclass
class PassInfo:
    """Metadata for a compiler pass."""

    fn: Callable
    is_joint: bool  # True if applied on joint graph, False if on fwd/bwd graphs
    supported_modes: set = field(default_factory=set)


PASS_REGISTRY: dict[str, PassInfo] = {
    "auto_bucketing": PassInfo(
        fn=autobucketing_reordering_pass,
        is_joint=False,
        supported_modes={"jit", "aot"},
    ),
    "transformer_block_bucketing": PassInfo(
        fn=transformer_block_bucketing_reordering_pass,
        is_joint=False,
        supported_modes={"jit", "aot"},
    ),
    "regional_inductor": PassInfo(
        fn=regional_inductor_pass,
        is_joint=False,
        supported_modes={"aot"},
    ),
    "cudagraph": PassInfo(
        fn=cudagraph_pass,
        is_joint=False,
        supported_modes={"aot"},
    ),
    "full_inductor_compilation": PassInfo(
        fn=full_inductor_compilation_pass,
        is_joint=False,
        supported_modes={"aot"},
    ),
    "inductor_decomposition": PassInfo(
        fn=inductor_decomposition_pass,
        is_joint=True,
        supported_modes={"aot"},
    ),
}


def _validate_pass_constraints(pass_names: list[str]) -> None:
    """Validate pass ordering and mutual exclusion constraints."""
    if "cudagraph" in pass_names:
        if pass_names[-1] != "cudagraph":
            raise ValueError("cudagraph must be the last pass in the list")

    if "auto_bucketing" in pass_names and "transformer_block_bucketing" in pass_names:
        raise ValueError(
            "Cannot apply auto_bucketing and transformer_block_bucketing at the same time"
        )

    if "full_inductor_compilation" in pass_names:
        if "inductor_decomposition" not in pass_names:
            raise ValueError(
                "full_inductor_compilation requires inductor_decomposition. "
                "Please add inductor_decomposition to compile.passes"
            )


def validate_and_get_passes(
    pass_names: list[str],
    mode: str,
    transformer_block_buckets: list | None = None,
) -> tuple[list[Callable], list[Callable]]:
    """
    Validate and split passes into joint and fwd/bwd lists.

    Args:
        pass_names: List of pass names from config
        mode: Compilation mode ("jit" or "aot")
        transformer_block_buckets: Bucket plans for transformer_block_bucketing pass

    Returns:
        (joint_passes, fwd_bwd_passes)

    Raises:
        ValueError: If a pass is unknown, unsupported in the given mode,
                    or constraints are violated
    """
    _validate_pass_constraints(pass_names)

    joint_passes = []
    fwd_bwd_passes = []

    for name in pass_names:
        if name not in PASS_REGISTRY:
            raise ValueError(
                f"Unknown pass: {name}. "
                f"Available passes: {list(PASS_REGISTRY.keys())}"
            )

        info = PASS_REGISTRY[name]
        if mode not in info.supported_modes:
            raise ValueError(
                f"Pass '{name}' is not supported in '{mode}' mode. "
                f"Supported modes: {info.supported_modes}"
            )

        fn = info.fn

        # Apply model-specific configuration
        if name == "transformer_block_bucketing":
            if transformer_block_buckets is None:
                raise ValueError(
                    "transformer_block_bucketing requires transformer_block_buckets"
                )
            fn = functools.partial(fn, fsdp_manual_buckets=transformer_block_buckets)

        if info.is_joint:
            joint_passes.append(fn)
        else:
            fwd_bwd_passes.append(fn)

    if pass_names:
        logger.info(f"Using compiler passes: {pass_names}")
        if "full_inductor_compilation" in pass_names:
            logger.warning(
                "Full Inductor compilation is enabled. Note that Inductor may change "
                "numerics and does not guarantee bitwise equivalent results compared "
                "to eager mode."
            )

    return joint_passes, fwd_bwd_passes
