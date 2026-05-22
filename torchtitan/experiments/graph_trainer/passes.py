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


def remove_identity_views(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Peephole pass that removes identity ``aten.view.default`` nodes.

    DTensor plumbing (e.g. ``_api.py:112`` / ``_api.py:229``) emits many
    ``view.default(x, list(x.shape))`` calls — structural no-ops that still
    cost FX/compile time and can confuse downstream fusion. Replace each
    such node with its input.

    Only eliminate when input and target shape can be proven equal without
    symbolic guards; SymInt shapes that can't be statically resolved are
    skipped.
    """
    view_target = torch.ops.aten.view.default
    num_removed = 0
    num_view_nodes = 0
    skipped_no_meta = 0
    skipped_symint = 0
    skipped_shape_mismatch = 0

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not view_target:
            continue
        num_view_nodes += 1

        if len(node.args) < 2:
            continue
        input_node = node.args[0]
        target_size = node.args[1]
        if not isinstance(input_node, torch.fx.Node):
            continue
        val = input_node.meta.get("val", None)
        if val is None or not hasattr(val, "shape"):
            skipped_no_meta += 1
            continue
        input_shape = val.shape
        if len(input_shape) != len(target_size):
            skipped_shape_mismatch += 1
            continue

        equal = True
        for s_in, s_target in zip(input_shape, target_size):
            # Both ints: easy.
            if isinstance(s_in, int) and isinstance(s_target, int):
                if s_in != s_target:
                    equal = False
                    break
                continue
            # Otherwise try identity comparison (same SymInt object) — safe
            # because identical SymNodes mean identical symbolic values
            # without introducing guards.
            if s_in is s_target:
                continue
            # Mixed / non-identical SymInt: skip this view to avoid guards.
            equal = False
            skipped_symint += 1
            break

        if not equal:
            if not isinstance(input_shape[0], int) or any(
                not isinstance(s, int) for s in input_shape
            ):
                # Already accounted for in skipped_symint when applicable.
                pass
            else:
                # Shape mismatch on concrete ints — real reshape, leave alone.
                pass
            continue

        node.replace_all_uses_with(input_node)
        gm.graph.erase_node(node)
        num_removed += 1

    if num_removed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

    logger.info(
        f"remove_identity_views: removed {num_removed}/{num_view_nodes} "
        f"aten.view.default nodes "
        f"(skipped: no_meta={skipped_no_meta}, symint={skipped_symint}, "
        f"shape_mismatch={skipped_shape_mismatch})"
    )
    return gm


_INT64_MAX = 9223372036854775807


def remove_identity_ops(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Combined peephole pass for residual no-op patterns left by AOT/DTensor.

    Extends :func:`remove_identity_views` to three more zero-cost patterns:

    1. **Identity slice**: ``aten.slice.Tensor(x, dim, 0, end, 1)`` where
       ``end >= x.shape[dim]`` (incl. ``INT64_MAX``) — replace with ``x``.
    2. **Double transpose**: ``aten.t.default(aten.t.default(x))`` cancels
       to ``x`` (2D tensors). Replace the outer ``t`` with the grandparent.
    3. **Identity dtype cast**: ``aten._to_copy.default(x, dtype=x.dtype,
       ...)`` with no other tensor-attribute change — replace with ``x``.

    These nodes are no-ops in eager but still anchor FX nodes that delay
    deallocation post-AOT. Order matters: slice → double-t → identity cast
    so each step exposes more candidates for the next.
    """
    slice_target = torch.ops.aten.slice.Tensor
    t_target = torch.ops.aten.t.default
    to_copy_target = torch.ops.aten._to_copy.default

    num_slice_total = 0
    num_slice_removed = 0
    num_slice_skipped_meta = 0
    num_slice_skipped_symint = 0

    num_t_total = 0
    num_t_removed = 0

    num_to_copy_total = 0
    num_to_copy_removed = 0
    num_to_copy_skipped_meta = 0

    # ----- Pass 1: identity slice ----------------------------------------
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not slice_target:
            continue
        num_slice_total += 1

        if len(node.args) < 4:
            continue
        input_node = node.args[0]
        dim_arg = node.args[1]
        start = node.args[2]
        end = node.args[3]
        step = node.args[4] if len(node.args) >= 5 else 1
        if step is None:
            step = 1

        if not isinstance(input_node, torch.fx.Node):
            continue
        if not (
            isinstance(dim_arg, int)
            and isinstance(start, int)
            and isinstance(end, int)
            and isinstance(step, int)
        ):
            continue
        if start != 0 or step != 1:
            continue

        val = input_node.meta.get("val", None)
        if val is None or not hasattr(val, "shape"):
            num_slice_skipped_meta += 1
            continue
        shape = val.shape
        if dim_arg < 0:
            dim_arg = dim_arg + len(shape)
        if not (0 <= dim_arg < len(shape)):
            continue
        dim_size = shape[dim_arg]
        if not isinstance(dim_size, int):
            num_slice_skipped_symint += 1
            continue
        # end == INT64_MAX (or any value >= dim_size) means full range.
        if not (end >= dim_size or end == _INT64_MAX):
            continue

        node.replace_all_uses_with(input_node)
        gm.graph.erase_node(node)
        num_slice_removed += 1

    # ----- Pass 2: double transpose --------------------------------------
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not t_target:
            continue
        num_t_total += 1

        if len(node.args) < 1:
            continue
        inner = node.args[0]
        if not isinstance(inner, torch.fx.Node):
            continue
        if inner.op != "call_function" or inner.target is not t_target:
            continue
        if len(inner.args) < 1:
            continue
        grandparent = inner.args[0]
        if not isinstance(grandparent, torch.fx.Node):
            continue

        node.replace_all_uses_with(grandparent)
        gm.graph.erase_node(node)
        num_t_removed += 1
        # Leave `inner` alone — DCE will remove it if it now has no users,
        # otherwise it's still needed by some other path.

    # ----- Pass 3: identity dtype cast -----------------------------------
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not to_copy_target:
            continue
        num_to_copy_total += 1

        if len(node.args) < 1:
            continue
        input_node = node.args[0]
        if not isinstance(input_node, torch.fx.Node):
            continue
        val = input_node.meta.get("val", None)
        if val is None or not hasattr(val, "dtype"):
            num_to_copy_skipped_meta += 1
            continue

        target_dtype = node.kwargs.get("dtype", None)
        if target_dtype is not None and target_dtype != val.dtype:
            continue

        # Bail if the cast also changes layout / device / pin_memory in any
        # meaningful way. Treat `None` and a value matching the input as
        # "no change". We only inspect the bound kwargs; missing kwargs are
        # implicitly identity.
        non_identity = False
        target_layout = node.kwargs.get("layout", None)
        if target_layout is not None:
            input_layout = getattr(val, "layout", None)
            if input_layout is not None and target_layout != input_layout:
                non_identity = True

        target_device = node.kwargs.get("device", None)
        if target_device is not None:
            input_device = getattr(val, "device", None)
            if input_device is not None and target_device != input_device:
                non_identity = True

        # pin_memory=True would actually allocate. Only safe if False/None.
        target_pin = node.kwargs.get("pin_memory", None)
        if target_pin:
            non_identity = True

        # memory_format that forces a contiguous re-layout could matter.
        # Only treat as identity if absent or torch.preserve_format.
        target_mf = node.kwargs.get("memory_format", None)
        if target_mf is not None and target_mf is not torch.preserve_format:
            non_identity = True

        if non_identity:
            continue

        node.replace_all_uses_with(input_node)
        gm.graph.erase_node(node)
        num_to_copy_removed += 1

    total_removed = num_slice_removed + num_t_removed + num_to_copy_removed
    if total_removed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

    logger.info(
        f"remove_identity_ops: removed "
        f"slice={num_slice_removed}/{num_slice_total} "
        f"(skipped: no_meta={num_slice_skipped_meta}, "
        f"symint={num_slice_skipped_symint}), "
        f"double_t={num_t_removed}/{num_t_total}, "
        f"identity_cast={num_to_copy_removed}/{num_to_copy_total} "
        f"(skipped: no_meta={num_to_copy_skipped_meta})"
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
        remove_identity_views,
        remove_identity_ops,
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
