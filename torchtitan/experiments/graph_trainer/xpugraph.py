# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
XPUGraph pass for the graph trainer.

This module provides an XPUGraph pass that can be applied to graph modules
during compilation.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._library.opaque_object import is_opaque_value
from torch.utils._ordered_set import OrderedSet

from torchtitan.config.function import Function
from torchtitan.tools.logging import logger


class _XPUGraphManager:
    """A manager to hold a shared graph pool, stream, and wrapper registry."""

    def __init__(self) -> None:
        self._initialized = False
        self._xpugraph_wrappers: list["XPUGraphWrapper"] = []
        self._teardown_called = False
        self.graph_pool: Any | None = None
        self.stream: Any | None = None

    def maybe_initialize(self) -> None:
        if self._initialized:
            return

        if self._teardown_called:
            raise RuntimeError("Cannot initialize XPUGraph after teardown")

        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPUGraph requires an available XPU device")

        required_apis = (
            "XPUGraph",
            "graph",
            "graph_pool_handle",
            "Stream",
            "current_stream",
            "stream",
            "synchronize",
        )
        missing = [name for name in required_apis if not hasattr(torch.xpu, name)]
        if missing:
            raise RuntimeError(
                "This PyTorch build does not provide the required XPUGraph APIs: "
                + ", ".join(f"torch.xpu.{name}" for name in missing)
            )

        # Create a global XPUGraph memory pool to allow memory reuse across
        # XPUGraphs.
        self.graph_pool = torch.xpu.graph_pool_handle()

        # Create a global XPU stream for graph capture. We need to use a single
        # stream for all allocations to the memory pool; otherwise, allocations
        # made on separate streams cannot be reused through the shared pool.
        self.stream = torch.xpu.Stream()
        self._initialized = True

    def register_wrapper(self, wrapper: "XPUGraphWrapper") -> None:
        if self._teardown_called:
            raise RuntimeError("Cannot register a new XPUGraph after teardown")
        self._xpugraph_wrappers.append(wrapper)

    def teardown(self) -> None:
        """Destroy all XPUGraphs and release the XPUGraph memory pool.

        Note [explicit XPUGraph teardown]
        XPUGraph can retain references to communication and runtime resources,
        which may prevent clean process-group teardown. Explicitly release the
        XPUGraph objects held by ``_XPUGraphManager`` and ``XPUGraphWrapper``.
        If XPUGraph is not used, this is a no-op.
        """
        if not self._initialized:
            return

        if self._teardown_called:
            logger.warning("xpugraph manager teardown called twice")
            return

        # Ensure no replay or capture is still using the graph objects.
        torch.xpu.synchronize()

        for wrapper in self._xpugraph_wrappers:
            wrapper.teardown()
        self._xpugraph_wrappers.clear()

        self.stream = None
        self.graph_pool = None
        self._teardown_called = True


_xg_manager = _XPUGraphManager()


def xpugraph_teardown() -> None:
    """Destroy all XPUGraphs and release the XPUGraph memory pool.

    See Note [explicit XPUGraph teardown] for more details.
    """
    _xg_manager.teardown()


# XPU does not currently expose the CUDA graph kernel-annotation APIs used by
# ``torch.cuda._graph_annotations``. Keep no-op equivalents so configuration
# code can use the same interface for CUDA and XPU without importing CUDA-only
# modules.
def get_xpugraph_annotations() -> dict[int, list]:
    """Return all kernel annotations accumulated across XPUGraph captures.

    XPU graph kernel annotation capture is not currently exposed, so this
    always returns an empty mapping.
    """
    return {}


def enable_xpugraph_annotations() -> None:
    """Enable kernel annotation capture on subsequent XPUGraph recordings.

    XPU graph kernel annotation capture is not currently exposed, so this is a
    no-op that logs a warning.
    """
    logger.warning(
        "XPUGraph kernel annotations are not currently supported; "
        "continuing without annotations."
    )


def xpugraph_annotate_trace_post_processor() -> Function.Config:
    """Return a no-op XPUGraph profiler trace post-processor.

    Attach this to ``Profiler.Config.trace_post_processor`` to preserve the
    same configuration shape as the CUDAGraph path. XPU graph kernel annotation
    metadata is not currently available, so the trace is left unchanged.
    """
    return Function.Config(fn=_xpugraph_annotate_trace_file)


def _xpugraph_annotate_trace_file(trace_path: str) -> None:
    """Post-process a profiler trace with XPUGraph kernel annotations.

    XPU graph kernel annotation metadata is not currently available, so this is
    a no-op.
    """
    del trace_path


def insert_kernel_annotations_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Insert kernel annotations at module boundaries in the FX graph.

    XPU graph kernel annotation capture is not currently exposed, so this pass
    returns the graph unchanged.
    """
    del example_inputs
    return gm


class XPUGraphWrapper:
    """Wrap a callable with XPUGraph.

    It warms up the callable, records XPUGraph, and replays XPUGraph during
    runtime. It also handles static input tensors, which are tensors whose
    addresses do not change across runs.

    Args:
        runnable: The callable to wrap with XPUGraph. This can be a
            ``torch.fx.GraphModule`` when used in an FX graph pass, or any
            callable when used in PyTorch eager mode.
        example_inputs: A list of example inputs to the callable.
        static_input_indices: Indices identifying static input tensors. Static
            inputs are tensors whose memory addresses remain constant across
            invocations. Common examples include model weights, buffers, and
            outputs from previously wrapped XPUGraph functions.
        should_check_address: Whether to verify static input tensor addresses
            at runtime. This should only be enabled for debugging purposes.
        tensor_input_indices: Indices of tensor-valued inputs. This is useful
            when the argument list also contains opaque values such as a
            ``DeviceMesh``.
    """

    def __init__(
        self,
        runnable: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: Sequence[int] | None = None,
        should_check_address: bool = False,
        tensor_input_indices: Sequence[int] | None = None,
    ) -> None:
        _xg_manager.maybe_initialize()
        _xg_manager.register_wrapper(self)

        self._runnable = runnable
        self._static_input_indices = OrderedSet(static_input_indices or [])

        if tensor_input_indices is not None:
            self._input_indices_to_copy = [
                i
                for i in tensor_input_indices
                if i not in self._static_input_indices
            ]
        else:
            self._input_indices_to_copy = [
                i
                for i, inp in enumerate(example_inputs)
                if isinstance(inp, torch.Tensor)
                and i not in self._static_input_indices
            ]

        self._xpugraph: Any | None = None
        self._has_warmup = False
        self._args: tuple[Any, ...] | None = None
        self._input_addresses: list[int | None] = []
        self._output: Any = None

        # Debug-only option that checks static input tensor addresses at
        # runtime.
        self._should_check_address = should_check_address
        self._gm = runnable if isinstance(runnable, torch.fx.GraphModule) else None

    def print_readable(self, *args, **kwargs):
        """Delegate to the inner GraphModule's ``print_readable`` method."""
        if self._gm is None:
            raise AssertionError(
                "print_readable requires a GraphModule runnable"
            )
        return self._gm.print_readable(*args, **kwargs)

    def _copy_non_static_inputs(self, *args: Any) -> None:
        if self._args is None:
            raise AssertionError(
                "XPUGraph input buffers have not been initialized"
            )

        for i in self._input_indices_to_copy:
            dst = self._args[i]
            src = args[i]

            if not isinstance(dst, torch.Tensor) or not isinstance(
                src, torch.Tensor
            ):
                raise TypeError(
                    f"Expected tensor input at index {i}, got "
                    f"{type(src).__name__} -> {type(dst).__name__}"
                )

            dst.copy_(src)

    def _validate_inputs(self, inputs: Sequence[Any]) -> None:
        """Validate that all inputs are of supported types.

        Opaque inputs, such as ``DeviceMesh`` values from SimpleFSDP/DTensor,
        are inherently static and already excluded from copying because only
        tensors appear in ``_input_indices_to_copy``. No special handling is
        needed beyond accepting them here.
        """
        for i, inp in enumerate(inputs):
            if isinstance(
                inp,
                (
                    torch.Tensor,
                    int,
                    float,
                    torch._C.Generator,
                ),
            ):
                continue

            if is_opaque_value(inp):
                continue

            raise ValueError(
                "args must be tensor, integer (for dynamic shapes), "
                "float (for scalar constants), Generator, or opaque object, "
                f"but found {type(inp)} with value {inp!r} at index {i}"
            )

    def _check_static_inputs_address(self) -> None:
        if self._args is None:
            raise AssertionError(
                "XPUGraph input buffers have not been initialized"
            )

        for i in self._static_input_indices:
            value = self._args[i]

            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Static input index {i} is not a tensor")

            actual = value.data_ptr()
            expected = self._input_addresses[i]

            if expected != actual:
                raise AssertionError(
                    "Expected the same static tensor address but found "
                    f"{expected} != {actual} at input index {i}"
                )

    def _warmup(self, *args: Any) -> Any:
        """Warm up the callable on the shared XPU capture stream."""
        capture_stream = _xg_manager.stream
        if capture_stream is None:
            raise AssertionError(
                "XPUGraph capture stream is not initialized"
            )

        current_stream = torch.xpu.current_stream()
        capture_stream.wait_stream(current_stream)

        # Warm up lazy kernels and communication libraries on the same side
        # stream that will later be used for XPUGraph capture.
        with torch.xpu.stream(capture_stream):
            out = self._runnable(*args)

        current_stream.wait_stream(capture_stream)
        return out

    def __call__(self, *args: Any) -> Any:
        if not self._has_warmup:
            self._has_warmup = True
            return self._warmup(*args)

        if self._xpugraph is None:
            self._validate_inputs(args)

            self._args = tuple(args)
            self._input_addresses = [
                value.data_ptr()
                if isinstance(value, torch.Tensor)
                else None
                for value in args
            ]

            graph_pool = _xg_manager.graph_pool
            capture_stream = _xg_manager.stream

            if graph_pool is None or capture_stream is None:
                raise AssertionError(
                    "XPUGraph manager is not initialized"
                )

            self._xpugraph = torch.xpu.XPUGraph()

            with torch.xpu.graph(
                self._xpugraph,
                pool=graph_pool,
                stream=capture_stream,
            ):
                # ``output`` is managed by PyTorch's XPUGraph memory pool.
                self._output = self._runnable(*args)

        if self._should_check_address:
            self._check_static_inputs_address()

        self._copy_non_static_inputs(*args)
        self._xpugraph.replay()
        return self._output

    def teardown(self) -> None:
        """Destroy XPUGraph and release references.

        See Note [explicit XPUGraph teardown] for more details.
        """
        if self._xpugraph is not None:
            reset = getattr(self._xpugraph, "reset", None)
            if reset is not None:
                reset()

        self._xpugraph = None
        self._args = None
        self._input_addresses = []
        self._output = None


def _has_dynamic_shape(val: Any) -> bool:
    """True if ``val`` contains a tensor with a symbolic shape.

    A symbolic dimension is represented by ``torch.SymInt`` rather than a
    concrete integer.
    """
    if isinstance(val, torch.Tensor):
        return any(
            isinstance(size, torch.SymInt)
            for size in val.shape
        )

    if isinstance(val, (list, tuple)):
        return any(_has_dynamic_shape(item) for item in val)

    return False


def _iter_tensors(val: Any) -> list[torch.Tensor]:
    """Flatten ``val`` (tensor/list/tuple) to the tensors it contains."""
    if isinstance(val, torch.Tensor):
        return [val]

    if isinstance(val, (list, tuple)):
        return [
            tensor
            for item in val
            for tensor in _iter_tensors(item)
        ]

    return []


def _is_xpu_collective_target(target: Any) -> bool:
    """Return True for c10d functional or direct c10d collective operators."""
    target_name = str(target)

    return target_name.startswith(
        (
            "_c10d_functional.",
            "c10d_functional.",
            "c10d.",
        )
    )


def _is_known_xpugraph_unsafe_target(target: Any) -> bool:
    """Return True for an operator that is unsafe for XPUGraph capture."""
    if _is_xpu_collective_target(target):
        # XCCL/c10d graph capture is not generally available. Keeping the
        # collective outside a whole-graph capture requires regional capture;
        # this file intentionally implements only all-or-nothing capture.
        return True

    if target == torch.ops.aten._local_scalar_dense.default:
        # ``.item()``/``.tolist()`` require a device-to-host synchronization
        # that an XPUGraph replay cannot repeat.
        return True

    if target == torch.ops.aten.index_put_.default:
        # This path can trigger an XPU event wait during graph capture.
        return True

    return False


def is_xpugraphable(
    node: torch.fx.Node,
    dyn_map: dict[torch.fx.Node, bool] | None = None,
) -> bool:
    """Whether ``node`` can be captured by an XPU graph.

    This is the per-node predicate used by the whole-graph build-time gate
    (:func:`is_full_xpugraphable`). ``dyn_map``, when provided, is a precomputed
    ``{node: has_dynamic_shape(output)}`` map, so each dynamic-shape check is an
    O(1) lookup instead of being recomputed for every consumer.
    """
    if node.op != "call_function":
        return True

    # ``getitem`` only indexes a multi-output operation's result, so it
    # inherits its parent's XPUGraph ability rather than being judged on its
    # own tensor metadata.
    if node.target is operator.getitem:
        parent = node.args[0]

        return (
            not isinstance(parent, torch.fx.Node)
            or is_xpugraphable(parent, dyn_map)
        )

    if _is_known_xpugraph_unsafe_target(node.target):
        return False

    # Cross-device copies from or to unpinned CPU memory are not replayable.
    # The CPU side must be pinned for an asynchronous H2D/D2H copy.
    if node.target in (
        torch.ops.aten.copy_.default,
        torch.ops.aten._to_copy.default,
    ):
        output_val = node.meta.get("val")

        if isinstance(output_val, torch.Tensor):
            for input_node in node.all_input_nodes:
                input_val = input_node.meta.get("val")

                if (
                    isinstance(input_val, torch.Tensor)
                    and input_val.device.type
                    != output_val.device.type
                ):
                    cpu_val = (
                        output_val
                        if output_val.device.type == "cpu"
                        else input_val
                    )

                    if (
                        cpu_val.device.type == "cpu"
                        and not cpu_val.is_pinned()
                    ):
                        return False

    # Reject an operation with a dynamic input or output shape.
    def _dyn(n: torch.fx.Node) -> bool:
        if dyn_map is not None:
            return dyn_map.get(n, False)

        return _has_dynamic_shape(n.meta.get("val"))

    if _dyn(node) or any(
        _dyn(inp) for inp in node.all_input_nodes
    ):
        return False

    tensors = _iter_tensors(node.meta.get("val"))

    for input_node in node.all_input_nodes:
        tensors.extend(
            _iter_tensors(input_node.meta.get("val"))
        )

    # An XPUGraph can contain XPU work and supported pinned CPU transfers.
    # Reject work targeting another accelerator backend.
    if any(
        tensor.device.type not in ("xpu", "cpu")
        for tensor in tensors
    ):
        return False

    # Pure-CPU operations execute on the host during capture and would replay
    # stale results because XPUGraph only records XPU device work.
    if tensors and all(
        tensor.device.type == "cpu"
        for tensor in tensors
    ):
        return False

    return True


def is_full_xpugraphable(
    gm: torch.fx.GraphModule,
) -> bool:
    """True if every node is XPUGraphable.

    A True result means the graph can be captured as one full XPU graph.
    """
    dyn_map = {
        node: _has_dynamic_shape(node.meta.get("val"))
        for node in gm.graph.nodes
    }

    return all(
        is_xpugraphable(node, dyn_map)
        for node in gm.graph.nodes
    )


def is_xpugraph_compatible(
    gm: torch.fx.GraphModule,
) -> bool:
    """Whole-graph XPUGraph gate.

    Returns True only when the graph has no XPUGraph-unsafe operation. A
    preceding pass can also store its pre-collapse compatibility verdict in
    ``gm.meta['xpugraph_compatible']`` when later graph transformations would
    otherwise hide the original operations from this scan.
    """
    if gm.meta.get("xpugraph_compatible") is False:
        logger.warning(
            "Skipping xpugraph: "
            "gm.meta['xpugraph_compatible'] is False."
        )
        return False

    return is_full_xpugraphable(gm)


def get_static_input_indices(
    gm: torch.fx.GraphModule,
    is_forward: bool,
) -> list[int]:
    """Get indices of graph inputs that are static input tensors.

    Static input tensor addresses do not change across runs. Examples include
    weights, buffers, and outputs of previously XPUGraph-wrapped functions.
    """
    from torch._inductor.utils import count_tangents

    static_input_indices: list[int] = []

    if (
        is_forward
        and (
            tracing_context
            := torch._guards.TracingContext.try_get()
        )
        and hasattr(tracing_context, "fw_metadata")
    ):
        # For forward, rely on graph capture, such as Dynamo or export, to
        # provide the correct static input indices in the tracing context.
        # Typical examples include weights and buffers.
        static_input_indices = list(
            tracing_context.fw_metadata.static_input_indices
        )

    elif not is_forward:
        # For backward, saved tensors are static inputs because they are outputs
        # of the XPUGraph-wrapped forward run. In a PT2-generated backward
        # GraphModule, saved tensors are the leading arguments. Count the saved
        # tensors and use their leading indices as the static input indices.
        fixed = count_tangents(gm)
        static_input_indices = list(range(fixed))

    return static_input_indices


def xpugraph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    is_forward: bool = True,
    static_input_indices: list[int] | None = None,
    tensor_input_indices: list[int] | None = None,
) -> torch.fx.GraphModule:
    """Apply XPUGraph.

    This pass wraps the forward function with XPUGraph during compilation and
    does not record XPUGraph until runtime.

    - On the first run, it warms up operators and runtime libraries.
    - On the second run, it records XPUGraph and replays XPUGraph.
    - On subsequent runs, it replays XPUGraph.

    Args:
        gm: The graph module to wrap.
        example_inputs: Example inputs used to determine tensor inputs for
            copying during replay.
        is_forward: Whether this is a forward graph (True) or backward graph
            (False). This is used to infer which inputs have stable tensor
            addresses when ``static_input_indices`` is not provided. It
            defaults to True because GraphTrainer traces one combined
            forward/loss/backward graph and wraps it as the forward graph.
        static_input_indices: Explicit input indices whose tensor addresses are
            stable. When provided, ``is_forward`` is not used for inference.
        tensor_input_indices: Indices of graph inputs that are tensors rather
            than opaque values such as ``DeviceMesh``. These indices determine
            which non-static inputs must be copied before replay. When omitted,
            they are inferred from ``example_inputs``.
    """
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            "xpugraph_pass requires a GraphModule but got "
            f"{type(gm).__name__}. Ensure xpugraph is not combined "
            "with a pass that replaces the GraphModule "
            "(for example, full_inductor_compilation)."
        )

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        logger.warning(
            "Skipping xpugraph: XPU is not available."
        )
        return gm

    if not is_xpugraph_compatible(gm):
        logger.warning(
            "Skipping xpugraph: graph is not compatible after all "
            "preceding passes. Use "
            "--compile.disable_passes xpugraph_pass to silence."
        )
        return gm

    if static_input_indices is None:
        static_input_indices = get_static_input_indices(
            gm,
            is_forward,
        )

    gm.forward = XPUGraphWrapper(
        gm.forward,
        example_inputs,
        static_input_indices,
        tensor_input_indices=tensor_input_indices,
    )

    logger.info("Applied xpugraph pass.")
    return gm
