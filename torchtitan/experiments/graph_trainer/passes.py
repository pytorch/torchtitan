# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph pass infrastructure for the autoresearch experiment.

This module provides the pass orchestration framework: building the pass list,
applying passes in order, and debug instrumentation. The agent adds custom
pass functions and registers them in ``construct_default_graph_passes``.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable

import torch
from torch._logging import trace_structured

from torchtitan.experiments.graph_trainer.debug_utils import (
    log_graph_diff,
    snapshot_graph,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.tools.logging import logger


def compile_time_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
    *,
    use_cudagraph: bool = False,
) -> list[Callable]:
    """Return compile-time passes (used by precompile path)."""
    return construct_default_graph_passes(traced_result, config)


def remove_detach_nodes(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
    """Remove ``aten.detach.default`` nodes from the joint fwd+bwd graph.

    The graph is executed inside a ``torch.no_grad()`` block at runtime, so
    ``detach`` has no autograd effect. Each ``detach.default`` call still
    allocates a new tensor view in the dispatcher, so removing them shaves
    a small amount of per-step launch and bookkeeping overhead.

    Strategy:
        1. Collect all ``aten.detach.default`` call_function nodes.
        2. Rewire every user of each node to read from ``node.args[0]``.
        3. Erase the now-orphaned detach nodes.
        4. Run ``eliminate_dead_code`` once to clean up any newly-dead ops,
           followed by ``graph.lint()`` and ``gm.recompile()``.
    """
    detach_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten.detach.default
    ]
    for node in detach_nodes:
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    logger.info(f"remove_detach_nodes: removed {len(detach_nodes)} detach nodes")
    return gm


def apply_inductor_pattern_passes(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
    """Apply Inductor's joint-graph FX pattern matchers to the traced graph.

    Inductor ships several FX-level pattern matchers that fuse common ATen
    sequences (mm+bias+activation, SDPA epilogues, _to_copy round-trips,
    binary folding, symint dedup, etc.). They operate in place on the
    GraphModule and don't require re-tracing. We try several upstream
    entry points, log what's available and what changes the graph, then
    lint+recompile.

    Each call is independently wrapped: import failures and runtime errors
    are logged and skipped, never re-raised.
    """

    def _node_count(g: torch.fx.GraphModule) -> int:
        return len(list(g.graph.nodes))

    initial_count = _node_count(gm)
    logger.info(f"apply_inductor_pattern_passes: initial node count {initial_count}")

    # 1) joint_graph_passes
    try:
        from torch._inductor.fx_passes.joint_graph import joint_graph_passes

        try:
            before = _node_count(gm)
            joint_graph_passes(gm)
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: joint_graph_passes ran "
                f"(nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: joint_graph_passes raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info("apply_inductor_pattern_passes: joint_graph_passes not available")

    # 2) post_grad_passes (training mode)
    try:
        from torch._inductor.fx_passes.post_grad import post_grad_passes

        try:
            before = _node_count(gm)
            post_grad_passes(gm, is_inference=False)
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: post_grad_passes(is_inference=False) "
                f"ran (nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: post_grad_passes raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info("apply_inductor_pattern_passes: post_grad_passes not available")

    # 3) pre_grad_passes
    try:
        from torch._inductor.fx_passes.pre_grad import pre_grad_passes

        try:
            before = _node_count(gm)
            pre_grad_passes(gm, example_inputs)
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: pre_grad_passes ran "
                f"(nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: pre_grad_passes raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info("apply_inductor_pattern_passes: pre_grad_passes not available")

    # 4) binary_folding_pass
    try:
        from torch._inductor.fx_passes.binary_folding import binary_folding_pass

        try:
            before = _node_count(gm)
            binary_folding_pass(gm.graph)
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: binary_folding_pass ran "
                f"(nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: binary_folding_pass raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info(
            "apply_inductor_pattern_passes: binary_folding_pass not available"
        )

    # 5) dedupe_symint_uses_pass
    try:
        from torch._inductor.fx_passes.dedupe_symint_uses import (
            dedupe_symint_uses_pass,
        )

        try:
            before = _node_count(gm)
            dedupe_symint_uses_pass(gm.graph)
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: dedupe_symint_uses_pass ran "
                f"(nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: dedupe_symint_uses_pass raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info(
            "apply_inductor_pattern_passes: dedupe_symint_uses_pass not available"
        )

    gm.graph.lint()
    gm.recompile()
    final_count = _node_count(gm)
    logger.info(
        f"apply_inductor_pattern_passes: final node count {final_count} "
        f"(delta {final_count - initial_count})"
    )
    return gm


def construct_default_graph_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
) -> list[Callable]:
    """Build the pass list for the aot_fx_trace path.

    The agent adds custom passes to this list.
    """
    passes: list[Callable] = [
        remove_detach_nodes,
        apply_inductor_pattern_passes,
    ]
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
    """Log the transformed graph to tlparse via trace_structured."""
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
