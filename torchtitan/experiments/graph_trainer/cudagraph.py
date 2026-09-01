# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CUDAGraph pass for the graph trainer.

This module provides a cudagraph pass that can be applied to graph modules
during compilation.
"""

import operator
from typing import Any

import torch

from torchtitan.distributed.cudagraph import CUDAGraphWrapper
from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
from torchtitan.tools.logging import logger


def _has_dynamic_shape(val: Any) -> bool:
    """True if ``val`` is (or contains) a tensor with a symbolic (data-dependent)
    shape — i.e. any dimension is a ``torch.SymInt`` rather than a concrete int."""
    if isinstance(val, torch.Tensor):
        return any(isinstance(s, torch.SymInt) for s in val.shape)
    if isinstance(val, (list, tuple)):
        return any(_has_dynamic_shape(v) for v in val)
    return False


def _iter_tensors(val: Any) -> list[torch.Tensor]:
    """Flatten ``val`` (tensor / list / tuple) to the tensors it contains."""
    if isinstance(val, torch.Tensor):
        return [val]
    if isinstance(val, (list, tuple)):
        return [t for v in val for t in _iter_tensors(v)]
    return []


def is_cudagraphable(
    node: torch.fx.Node, dyn_map: dict[torch.fx.Node, bool] | None = None
) -> bool:
    """Whether ``node`` can be captured by a CUDA graph.

    Per-node predicate for the partitioner (:func:`cudagraph_pass`) and the
    build-time gate (:func:`is_full_cudagraphable`). ``dyn_map``, when given, is a
    precomputed ``{node: has_dynamic_shape(out)}`` map so the shape check is an
    O(1) lookup instead of recomputing per consumer.

    flex_attention HOPs count as cudagraphable: regional_inductor compiles them to
    Triton kernels before cudagraph, so this never sees a flex HOP at capture time.
    """
    if node.op != "call_function":
        return True

    # getitem only indexes a multi-output op's result, so it inherits its parent's
    # cudagraph-ability rather than being judged on its own tensors (e.g. a getitem
    # of an eager dynamic-shape/CPU op must also be eager).
    if node.target is operator.getitem:
        parent = node.args[0]
        return not isinstance(parent, torch.fx.Node) or is_cudagraphable(parent)

    # Cross-device copy from/to unpinned CPU memory: cudagraph requires the CPU
    # side to be pinned for the async H2D/D2H copy.
    if node.target in (
        torch.ops.aten.copy_.default,
        torch.ops.aten._to_copy.default,
    ):
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            for inp in node.all_input_nodes:
                inp_val = inp.meta.get("val")
                if (
                    isinstance(inp_val, torch.Tensor)
                    and inp_val.device.type != val.device.type
                ):
                    cpu_val = val if val.device.type == "cpu" else inp_val
                    if not cpu_val.is_pinned():
                        return False

    # aten._grouped_mm may perform internal CPU<->CUDA copies not visible in FX
    # metadata; resolved on sm_100+.
    if node.target == torch.ops.aten._grouped_mm.default:
        if torch.cuda.get_device_capability() < (10, 0):
            return False

    # .item()/.tolist() need a device-to-host sync a cudagraph replay can't redo.
    if node.target == torch.ops.aten._local_scalar_dense.default:
        return False

    # Op with a dynamic (data-dependent / unbacked-SymInt) input or output shape.
    def _dyn(n: torch.fx.Node) -> bool:
        if dyn_map is not None:
            return dyn_map.get(n, False)
        return _has_dynamic_shape(n.meta.get("val"))

    if _dyn(node) or any(_dyn(inp) for inp in node.all_input_nodes):
        return False

    # Pure-CPU op: every tensor in its inputs and output lives on CPU. A CUDA graph
    # only captures CUDA kernels -- a CPU op would run on the host at capture time
    # and replay stale (e.g. the .tolist() unbind/getitem on EP token-routing split
    # sizes). Run it eager.
    tensors = _iter_tensors(node.meta.get("val"))
    for inp in node.all_input_nodes:
        tensors += _iter_tensors(inp.meta.get("val"))
    if tensors and all(t.device.type == "cpu" for t in tensors):
        return False

    return True


def is_full_cudagraphable(gm: torch.fx.GraphModule) -> bool:
    """True if every node is cudagraphable (:func:`is_cudagraphable`), i.e. the
    graph can be captured as one full CUDA graph. Used to resolve
    ``cudagraph_mode='auto'`` to full vs piecewise at pipeline-build time. Run on
    the pre-inductor graph; flex counts as cudagraphable (compiled before capture),
    matching the post-inductor reality."""
    return all(is_cudagraphable(node) for node in gm.graph.nodes)


def is_cudagraph_compatible(
    gm: torch.fx.GraphModule,
    *,
    skip_flex_attention_check: bool = False,
) -> bool:
    """Whole-graph cudagraph gate: True iff the graph has no cudagraph-unsafe op.

    Used by the all-or-nothing :func:`cudagraph_pass` and by
    ``full_inductor_compilation_pass`` (which stashes its pre-collapse verdict in
    ``gm.meta`` -- the collapse hides ops from the scan). Delegates to the per-node
    predicate via :func:`is_full_cudagraphable`.

    TODO: ``skip_flex_attention_check`` is now a no-op (flex_attention HOPs are no
    longer flagged -- regional_inductor compiles them before cudagraph). Remove the
    arg (and, once the all-or-nothing path is gone, this whole function) in a later
    PR.
    """
    if gm.meta.get("cudagraph_compatible") is False:
        logger.warning(
            "Skipping cudagraph: gm.meta['cudagraph_compatible'] is False "
            "(set by full_inductor_compilation_pass before the collapse)."
        )
        return False
    return is_full_cudagraphable(gm)


def get_static_input_indices(gm: torch.fx.GraphModule, is_forward: bool) -> list[int]:
    """
    Get indices of gm inputs that are static input tensors whose tensor addresses do not
    change across runs. Example of static input tensors include weights, buffers, and
    outputs of previous cudagraph wrapped functions.
    """
    from torch._inductor.utils import count_tangents

    static_input_indices = []
    if (
        is_forward
        and (tracing_context := torch._guards.TracingContext.try_get())
        and hasattr(tracing_context, "fw_metadata")
    ):
        # for forward, we rely on graph capture (i.e., dynamo or export) to provide
        # the correct static input indices stored in tracing context. Typical examples
        # include weights and buffers.
        static_input_indices = tracing_context.fw_metadata.static_input_indices

    elif not is_forward:
        # for backward, we identify saved tensors as static inputs, since saved tensors
        # are outputs of cudagraph-wrapped forward run. In PT2-generated backward gm,
        # saved tensors are always the leading args. So we can get the number of saved
        # tensors and generate static input indices.
        fixed = count_tangents(gm)
        static_input_indices = list(range(fixed))

    return static_input_indices


def insert_kernel_annotations_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Insert mark_kernels() calls at module boundaries in the FX graph.

    Reads ``node.meta["custom"]["module_fqn"]`` (set via
    ``annotate_module_fqns``) and inserts enter/exit calls so that
    CUDA graph capture records the annotations.

    Requires ``cuda-python`` package and CUDA toolkit/driver >= 13.1
    (or cuda-compat >= 13.1).  Returns the graph unchanged when unavailable.

    Alternative approaches:

    1. **fx.Interpreter**: During cudagraph capture, run the graph via an
       ``fx.Interpreter`` subclass that reads ``module_fqn`` metadata and
       calls ``mark_kernels`` enter/exit around each node — avoids mutating
       the graph.
    2. **Custom CodeGen**: Use a custom ``torch.fx.graph.CodeGen`` to emit
       enter/exit lines (or ``with`` blocks) directly in the generated
       Python code.

    The current graph-pass approach is the least invasive.
    """
    from torch.cuda._graph_annotations import _is_tools_id_unavailable

    def _enter(annotation: dict) -> object:
        from torch.cuda._graph_annotations import mark_kernels

        ctx = mark_kernels(annotation)
        ctx.__enter__()
        return ctx

    def _exit(ctx: object) -> None:
        ctx.__exit__(None, None, None)  # type: ignore[union-attr]

    if _is_tools_id_unavailable():
        return gm

    graph = gm.graph
    current_fqn: str | None = None
    current_ctx_node = None

    for node in list(graph.nodes):
        fqn = (node.meta.get("custom") or {}).get(_MODULE_FQN)

        if fqn != current_fqn:
            # Close previous scope
            if current_ctx_node is not None:
                with graph.inserting_before(node):
                    exit_node = graph.call_function(_exit, (current_ctx_node,))
                    exit_node.meta["custom"] = {}
                current_ctx_node = None

            # Open new scope
            if fqn is not None:
                with graph.inserting_before(node):
                    enter_node = graph.call_function(
                        _enter,
                        ({_MODULE_FQN: fqn},),
                    )
                    enter_node.meta["custom"] = {}
                current_ctx_node = enter_node

            current_fqn = fqn

    # Close any trailing scope (before output/return)
    if current_ctx_node is not None:
        output_nodes = [n for n in graph.nodes if n.op == "output"]
        if output_nodes:
            with graph.inserting_before(output_nodes[0]):
                exit_node = graph.call_function(_exit, (current_ctx_node,))
                exit_node.meta["custom"] = {}

    graph.lint()
    gm.recompile()
    return gm


def cudagraph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    is_forward: bool = True,
    static_input_indices: list[int] | None = None,
    tensor_input_indices: list[int] | None = None,
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
            when ``static_input_indices`` is not provided. Defaults to True --
            graph_trainer traces a single fwd+loss+bwd graph and always wraps it
            as the forward.
        static_input_indices: Explicit list of input indices with stable tensor
            addresses. When provided, ``is_forward`` is not used for inference.
        tensor_input_indices: Indices of graph inputs that are tensors (as
            opposed to opaque values like DeviceMesh). Used to compute which
            inputs need copying for cudagraph replay. When not provided, this
            is inferred from ``example_inputs``.
    """
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            f"cudagraph_pass requires a GraphModule but got {type(gm).__name__}. "
            f"Ensure cudagraph is not combined with passes that replace the "
            f"GraphModule (e.g. full_inductor_compilation)."
        )

    if not is_cudagraph_compatible(gm):
        logger.warning(
            "Skipping cudagraph: graph is not compatible after all preceding "
            "passes. Use --compile.disable_passes cudagraph_pass to silence."
        )
        return gm

    if static_input_indices is None:
        static_input_indices = get_static_input_indices(gm, is_forward)
    gm.forward = CUDAGraphWrapper(
        gm.forward,
        example_inputs,
        static_input_indices,
        tensor_input_indices=tensor_input_indices,
    )
    logger.info("Applied cudagraph pass.")
    return gm
