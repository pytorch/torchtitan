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

from __future__ import annotations

import functools
import operator
from collections import defaultdict
from collections.abc import Callable

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
from torchtitan.experiments.graph_trainer.custom_codegen import custom_codegen_pass
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_detach_pass,
    remove_identity_slice_pass,
    remove_identity_view_pass,
)
from torchtitan.experiments.graph_trainer.reshard_after_forward import (
    annotate_fsdp_all_gather,
)
from torchtitan.tools.logging import logger


def construct_default_graph_passes(
    traced_result: "TracedResult",
) -> list[Callable]:
    """Build the default pass list for the aot_fx_trace compile path.

    Per-pass configuration (e.g. ``static_input_indices`` for cudagraph) is
    bound here via ``functools.partial`` so that ``apply_graph_passes``
    stays a generic pass runner with no pass-specific parameters.

    Args:
        traced_result: The traced graph and metadata from ``trace_train_step``.

    Returns:
        An ordered list of graph passes ready to apply.
    """
    from torchtitan.models.common.attention import FlexAttention

    passes: list[Callable] = [
        functools.partial(tlparse_log_graph_pass, graph_name="make_fx_graph_traced"),
        remove_detach_pass,
        remove_identity_view_pass,
        remove_identity_slice_pass,
        # FlexAttention HOPs must be compiled (via regional_inductor) to
        # produce bitwise identical results to the eager Trainer path.
        # When left uncompiled, flex_attention still runs correctly but
        # produces different numerical results.
        functools.partial(
            annotate_flex_attention_for_regional_inductor_pass,
            flex_compile_config=FlexAttention.inductor_configs,
        ),
        regional_inductor_pass,
        # TODO: Switch to upstream PyTorch implementation when
        # https://github.com/pytorch/pytorch/pull/178246 lands.
        # custom_codegen_pass saves the FX graph to disk for:
        # 1. Debugging: inspect the generated graph code directly
        # 2. Profiling provenance: dual-path codegen with _RecordFunctionFast
        #    gives fine-grained operator-level attribution in profiler traces
        # 3. User-editable codegen: users can directly modify the generated
        #    program on disk for fine-grain scheduling optimizations, with
        #    hot-reload picking up changes at runtime
        # TODO: Investigate why custom_codegen_pass hangs when dumping codegen file
        # custom_codegen_pass,
    ]

    # cudagraph should be the last pass.
    from torchtitan.experiments.graph_trainer.cudagraph import is_cudagraph_compatible

    if is_cudagraph_compatible(traced_result.gm):
        static_input_indices = list(range(traced_result.num_static_inputs))
        passes.append(
            functools.partial(
                cudagraph_pass,
                is_forward=True,
                static_input_indices=static_input_indices,
            )
        )

    return passes


def apply_graph_passes(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    passes: list[Callable],
) -> torch.fx.GraphModule:
    """Apply graph passes to the traced fwd+bwd graph.

    Args:
        gm: The traced forward+backward graph module.
        example_inputs: Example (fake) inputs matching the graph signature.
        passes: Ordered list of pass callables, each with signature
            ``(gm, example_inputs, **kwargs) -> gm``.
    """
    for pass_fn in passes:
        gm = pass_fn(gm, example_inputs)
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
    allowed for the graph to serialize correctly.  The device_mesh ops
    (e.g. _get_submesh) appear in the backward graph when DTensor
    reconstructs submeshes from tracked ancestor meshes.
    """
    return name.startswith(
        (
            "torch.ops.aten",
            "torch.ops.fbgemm",
            "torch.ops._c10d_functional",
            "torch.ops._dtensor",
            "torch.ops.device_mesh",
        )
    )


def _node_metadata_key_filter_distributed(key: str) -> bool:
    """Metadata key filter for regional_inductor with distributed ops.

    Distributed ops (e.g. _get_submesh, mesh_get_process_group) produce
    opaque values (DeviceMesh, ProcessGroup) in node.meta["val"] and
    node.meta["eager_input_vals"] that cannot be pickled.  We strip
    both — they are not needed at runtime.
    """
    if key in ("val", "eager_input_vals"):
        return False
    return key not in ["source_fn_stack", "nn_module_stack", "fwd_source_fn_stack"]


def regional_inductor_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple, *, serializable: bool = False
) -> torch.fx.GraphModule:
    """Compile tagged graph regions with ``regional_inductor``.

    Scans the graph for nodes whose ``node.meta["custom"]`` contains a
    ``compile_with_inductor`` key and compiles those regions with
    TorchInductor.  Nodes without this tag are left unchanged.  If no
    nodes are tagged the pass is a no-op.

    Inductor is configured for bitwise-equal numerics so that the
    compiled regions match eager execution exactly.

    Args:
        gm: The graph module to compile.
        example_inputs: Example inputs for shape propagation.
        serializable: When True (precompile mode), sets
            ``force_autograd_cache`` so that ``regional_inductor`` wraps
            its output in ``RegionalOutputCode``, and overrides the ops
            filter to allow distributed collective ops.
    """
    import torch._inductor.config as ic
    from torch._subclasses.fake_tensor import FakeTensor

    def _get_fake_mode_from_gm(gm: torch.fx.GraphModule):
        """Extract the FakeTensorMode from a graph module's placeholder metadata."""
        for node in gm.graph.nodes:
            if node.op == "placeholder" and "val" in node.meta:
                val = node.meta["val"]
                if isinstance(val, FakeTensor):
                    return val.fake_mode
        return None

    # Ensure inductor produces bitwise-equal numerics vs eager.
    ic.eager_numerics.division_rounding = True
    # Recommended by inductor team — uncomment as needed:
    # ic.emulate_precision_casts = True
    # ic.eager_numerics.disable_ftz = True
    # ic.eager_numerics.use_pytorch_libdevice = True
    # ic.fallback_random = True

    # regional_inductor calls standalone_compile with
    # dynamic_shapes="from_tracing_context", which requires an active
    # TracingContext with a FakeTensorMode.  When this pass is called
    # outside torch.compile (e.g. after make_fx tracing in graph_trainer),
    # no TracingContext exists, so we create one from the graph's fake
    # tensor metadata.
    fake_mode = _get_fake_mode_from_gm(gm)
    tracing_ctx = torch._guards.TracingContext(fake_mode)

    if serializable:
        with (
            torch._guards.tracing(tracing_ctx),
            torch._functorch.config.patch("force_autograd_cache", True),
        ):
            result = regional_inductor(gm, example_inputs)
        from torch._inductor.output_code import RegionalOutputCode

        # Override the ops filter after compilation so that
        # serialization (which happens later) allows distributed
        # collective ops like _c10d_functional through GraphPickler.
        if isinstance(result, RegionalOutputCode):
            result._ops_filter = _ops_filter_with_distributed
            result._node_metadata_key_filter = _node_metadata_key_filter_distributed
        else:
            logger.warning(
                "regional_inductor with serializable=True did not produce "
                "RegionalOutputCode; distributed ops may not serialize correctly."
            )
        return result

    with torch._guards.tracing(tracing_ctx):
        gm = regional_inductor(gm, example_inputs)

    # regional_inductor may switch to boxed calling convention; reset to
    # default so the graph can be called with positional args as usual.
    gm.graph.set_codegen(torch.fx.graph.CodeGen())
    gm.recompile()
    return gm


def cudagraph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    is_forward: bool,
    static_input_indices: list[int] | None = None,
) -> torch.fx.GraphModule:
    """
    Apply cudagraph.

    This pass wraps the forward function with cudagraph during compilation and does
    not record cudagraph until runtime.
    - For the first run, it will warm up operators such as nccl.
    - For the second run, it will record cudagraph and replay cudagraph.
    - For the following runs, it will replay cudagraph.

    Args:
        gm: The graph module to wrap.
        example_inputs: Example inputs for warmup/recording.
        is_forward: Whether this is a forward graph (True) or backward graph
            (False). Used to infer which inputs have stable tensor addresses
            when ``static_input_indices`` is not provided.
        static_input_indices: Explicit list of input indices with stable tensor
            addresses. When provided, ``is_forward`` is not used for inference.
    """
    # Lazy import: cudagraph.py runs init_global_graph_pool() at import time,
    # which must happen after torch.cuda.set_device(local_rank).
    from torchtitan.experiments.graph_trainer.cudagraph import (
        CUDAGraphWrapper,
        get_static_input_indices,
    )

    if static_input_indices is None:
        static_input_indices = get_static_input_indices(gm, is_forward)
    gm.forward = CUDAGraphWrapper(gm.forward, example_inputs, static_input_indices)
    return gm


def annotate_flex_attention_for_regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    flex_compile_config: dict | None,
    mask_compile_config: dict | None = None,
) -> torch.fx.GraphModule:
    """Tag flex attention HOPs with compile_with_inductor for regional_inductor.

    Annotates three sets of nodes so that regional_inductor correctly
    scoops and compiles flex attention regions:
    1. The HOP node itself (flex_attention / flex_attention_backward)
    2. The get_attr nodes referencing score_mod / mask_mod submodules.
    3. All nodes inside those submodule graphs.

    Args:
        gm: The graph module to annotate.
        example_inputs: Example inputs (unused, required by pass interface).
        flex_compile_config: Inductor config dict for flex attention HOP
            nodes and their get_attr submodule references. When provided,
            wrapped as ``{"inductor_configs": flex_compile_config}``.
            When None, nodes are tagged with an empty annotation.
        mask_compile_config: Inductor config dict for nodes inside mask_mod
            subgraphs. When provided, wrapped as
            ``{"inductor_configs": mask_compile_config}``.
            When None, nodes are tagged with an empty annotation.
    """
    flex_compile_annotation: dict = (
        {"inductor_configs": flex_compile_config}
        if flex_compile_config is not None
        else {}
    )
    mask_compile_annotation: dict = (
        {"inductor_configs": mask_compile_config}
        if mask_compile_config is not None
        else {}
    )

    for node in gm.graph.nodes:
        if node.target not in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            continue
        node.meta.setdefault("custom", {})[
            "compile_with_inductor"
        ] = flex_compile_annotation
        for inp in node.all_input_nodes:
            if inp.op != "get_attr":
                continue
            submod = getattr(gm, inp.target, None)
            if not isinstance(submod, torch.fx.GraphModule):
                continue
            inp.meta.setdefault("custom", {})[
                "compile_with_inductor"
            ] = flex_compile_annotation

            # Following are the nodes in mask_mod subgraph
            for sub_node in submod.graph.nodes:
                sub_node.meta.setdefault("custom", {})[
                    "compile_with_inductor"
                ] = mask_compile_annotation
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
    "custom_codegen": custom_codegen_pass,
    "cudagraph": cudagraph_pass,
    "full_inductor_compilation": full_inductor_compilation_pass,
}

# Registry for joint custom passes (applied before partitioning, AOT mode only)
AVAILABLE_JOINT_PASSES = {
    "inductor_decomposition": inductor_decomposition_pass,
    "fsdp_reshard_after_fwd": fsdp_reshard_after_fwd_pass,
    "apply_sac": apply_sac_pass,
}
