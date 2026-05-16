# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
XPUGraph pass for the graph trainer.

This module provides an xpugraph pass that can be applied to graph modules
during compilation.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._library.opaque_object import is_opaque_value
from torch.utils._ordered_set import OrderedSet

from torchtitan.tools.logging import logger


def _get_xpu_graph_cls():
    if hasattr(torch.xpu, "XPUGraph"):
        return torch.xpu.XPUGraph
    if hasattr(torch.xpu, "graphs") and hasattr(torch.xpu.graphs, "XPUGraph"):
        return torch.xpu.graphs.XPUGraph
    raise RuntimeError("torch.xpu.XPUGraph is not available in this PyTorch build")


def _xpu_graph(*args, **kwargs):
    if hasattr(torch.xpu, "graph"):
        return torch.xpu.graph(*args, **kwargs)
    if hasattr(torch.xpu, "graphs") and hasattr(torch.xpu.graphs, "graph"):
        return torch.xpu.graphs.graph(*args, **kwargs)
    raise RuntimeError("torch.xpu.graph is not available in this PyTorch build")


def _xpu_graph_pool_handle():
    if hasattr(torch.xpu, "graph_pool_handle"):
        return torch.xpu.graph_pool_handle()
    if hasattr(torch.xpu, "graphs") and hasattr(torch.xpu.graphs, "graph_pool_handle"):
        return torch.xpu.graphs.graph_pool_handle()
    raise RuntimeError(
        "torch.xpu.graph_pool_handle is not available in this PyTorch build"
    )


class _XPUGraphManager:
    """A manager to hold a shared graph pool, stream, and wrapper registry."""

    def __init__(self) -> None:
        self._initialized = False
        self._xpugraph_wrappers: list["XPUGraphWrapper"] = []
        self._teardown_called = False

    def maybe_initialize(self) -> None:
        if self._initialized:
            return

        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPUGraph requires torch.xpu to be available")

        self._initialized = True

        # Create a global xpugraph memory pool to allow memory reuse across xpugraphs.
        self.graph_pool = _xpu_graph_pool_handle()

        # Create a global XPU stream for graph capture. Use one stream for all
        # allocations to the graph pool so allocations can be reused consistently.
        self.stream = torch.xpu.Stream()

        # Use a dummy graph to keep the global graph pool alive.
        xpu_graph_cls = _get_xpu_graph_cls()
        self._dummy_graph = xpu_graph_cls()
        with (
            # Suppress an empty graph warning, since we intentionally create
            # an empty graph here.
            warnings.catch_warnings(record=True),
            _xpu_graph(
                self._dummy_graph,
                pool=self.graph_pool,
                stream=self.stream,
            ),
        ):
            pass

    def register_wrapper(self, wrapper: "XPUGraphWrapper") -> None:
        assert not self._teardown_called, "Cannot register new xpugraph after teardown"
        self._xpugraph_wrappers.append(wrapper)

    def teardown(self) -> None:
        """Destroy all xpugraphs and release the xpugraph memory pool.

        This mirrors the explicit cudagraph teardown pattern. If xpugraphs hold
        references to backend communication/runtime resources, explicit teardown
        avoids keeping those resources alive during process-group destruction.
        If xpugraph is not used, this is a no-op.
        """
        if not self._initialized:
            return
        if self._teardown_called:
            logger.warning("xpugraph manager teardown called twice")
            return

        for wrapper in self._xpugraph_wrappers:
            wrapper.teardown()
        self._xpugraph_wrappers.clear()

        self._dummy_graph = None
        self.stream = None
        self.graph_pool = None
        self._teardown_called = True


_xg_manager = _XPUGraphManager()


def xpugraph_teardown() -> None:
    """Destroy all xpugraphs and release the xpugraph memory pool."""
    _xg_manager.teardown()


class XPUGraphWrapper:
    """Wraps a callable with xpugraph.

    It warms up the callable, records an XPU graph, and replays the XPU graph
    during runtime. It also handles static input tensors, which are tensors
    whose tensor addresses do not change across runs.

    Args:
        runnable: The callable to wrap with XPU graph. This can be a
            torch.fx.GraphModule when used in an FX graph pass, or any callable
            when used in PyTorch eager mode.
        example_inputs: A list of example inputs to the callable.
        static_input_indices: Indices identifying static input tensors. Static
            inputs are tensors whose memory addresses remain constant across
            invocations. Common examples include model weights, buffers, and
            outputs from previously wrapped graph functions.
        should_check_address: Whether to verify static input tensor addresses
            at runtime. This should only be enabled for debugging.
        tensor_input_indices: Indices of graph inputs that are tensors, as
            opposed to opaque values like DeviceMesh.
    """

    def __init__(
        self,
        runnable: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: tuple[int] | list[int] | None = None,
        should_check_address: bool = False,
        tensor_input_indices: list[int] | None = None,
    ):
        _xg_manager.maybe_initialize()
        _xg_manager.register_wrapper(self)

        self._runnable = runnable
        self._static_input_indices = OrderedSet(
            static_input_indices if static_input_indices is not None else []
        )

        if tensor_input_indices is not None:
            self._input_indices_to_copy = [
                i for i in tensor_input_indices if i not in self._static_input_indices
            ]
        else:
            self._input_indices_to_copy = [
                i
                for i, inp in enumerate(example_inputs)
                if isinstance(inp, torch.Tensor) and i not in self._static_input_indices
            ]

        self._xpugraph = None
        self._has_warmup = False

        self._args = None
        self._output = None

        # Debug only: whether to check static input tensor addresses at runtime.
        self._should_check_address = should_check_address

        self._gm = runnable if isinstance(runnable, torch.fx.GraphModule) else None

    def print_readable(self, *args, **kwargs):
        """Delegate to the inner GraphModule's print_readable."""
        assert self._gm is not None, "print_readable requires a GraphModule runnable"
        return self._gm.print_readable(*args, **kwargs)

    def _copy_non_static_inputs(self, *args):
        for i in self._input_indices_to_copy:
            self._args[i].copy_(args[i])

    def _validate_inputs(self, inputs) -> None:
        """Validate that all inputs are of supported types.

        Opaque inputs, such as DeviceMesh from SimpleFSDP/DTensor, are
        inherently static and already excluded from copying because only tensor
        inputs appear in ``_input_indices_to_copy``.
        """
        for i, inp in enumerate(inputs):
            if isinstance(inp, (torch.Tensor, int, float, torch._C.Generator)):
                continue
            if is_opaque_value(inp):
                continue
            raise ValueError(
                "args must be tensor, integer for dynamic shapes, "
                "float for scalar constants, "
                "Generator for random number generator, "
                "or opaque object, "
                f"but found {type(inp)} with value {inp!r} at index {i}"
            )

    def _check_static_inputs_address(self) -> None:
        for i in self._static_input_indices:
            actual = self._args[i].data_ptr()
            expected = self._input_addresses[i]
            assert expected == actual, (
                "Expected the same static tensor address but found "
                f"{expected} != {actual}"
            )

    def __call__(self, *args):
        # First run: warm up operators and runtime state normally.
        #
        # Unlike the CUDA version, this first version does not use
        # _use_cuda_memory_pool_manager because that helper is CUDA-specific.
        if not self._has_warmup:
            self._has_warmup = True
            return self._runnable(*args)

        # Second run: record the XPU graph.
        if self._xpugraph is None:
            self._validate_inputs(args)
            self._args = args
            self._input_addresses = [
                x.data_ptr() if isinstance(x, torch.Tensor) else None for x in args
            ]

            xpu_graph_cls = _get_xpu_graph_cls()
            self._xpugraph = xpu_graph_cls()

            with _xpu_graph(
                self._xpugraph,
                pool=_xg_manager.graph_pool,
                stream=_xg_manager.stream,
            ):
                # `output` is managed by PyTorch's graph pool.
                self._output = self._runnable(*args)

        if self._should_check_address:
            self._check_static_inputs_address()

        # Later runs: copy dynamic tensor inputs into the captured input buffers
        # and replay the graph.
        self._copy_non_static_inputs(*args)
        self._xpugraph.replay()
        return self._output

    def teardown(self) -> None:
        """Destroy xpugraph and release references."""
        self._xpugraph = None
        self._args = None
        self._output = None

_XPUGRAPH_UNSUPPORTED_OPS = {
    torch.ops.aten.index_put_.default: (
        "aten.index_put_ currently triggers an XPU event wait during "
        "command graph capture"
    ),
}

_FLEX_ATTENTION_OPS = {
    torch.ops.higher_order.flex_attention,
    torch.ops.higher_order.flex_attention_backward,
}


def is_xpugraph_compatible(gm: torch.fx.GraphModule) -> bool:
    """Check whether the graph can be safely captured by XPU graph.

    This is a conservative compatibility check. Known XPU command-graph
    incompatible ops are skipped before runtime capture so training does not
    fail inside torch.xpu.graph().
    """
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        reason = _XPUGRAPH_UNSUPPORTED_OPS.get(node.target)
        if reason is not None:
            logger.warning("Skipping xpugraph: %s", reason)
            return False

        # Check for aten.copy_ / aten._to_copy between CPU and XPU.
        if node.target in (
            torch.ops.aten.copy_.default,
            torch.ops.aten._to_copy.default,
        ):
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue

            for inp in node.all_input_nodes:
                inp_val = inp.meta.get("val")
                if (
                    isinstance(inp_val, torch.Tensor)
                    and inp_val.device.type != val.device.type
                ):
                    logger.warning(
                        "Skipping xpugraph: graph contains CPU↔XPU copy "
                        f"({node.target})"
                    )
                    return False

    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in _FLEX_ATTENTION_OPS:
            logger.warning(
                "Skipping xpugraph: graph contains flex_attention higher-order "
                "ops that require regional_inductor to compile before xpugraph "
                "can capture"
            )
            return False

    return True


def get_static_input_indices(gm: torch.fx.GraphModule, is_forward: bool) -> list[int]:
    """
    Get indices of gm inputs that are static input tensors whose tensor addresses
    do not change across runs.

    Example static input tensors include weights, buffers, and outputs of
    previous graph-wrapped functions.
    """
    from torch._inductor.utils import count_tangents

    static_input_indices = []
    if (
        is_forward
        and (tracing_context := torch._guards.TracingContext.try_get())
        and hasattr(tracing_context, "fw_metadata")
    ):
        # For forward, rely on graph capture, meaning Dynamo/export, to provide
        # the correct static input indices stored in tracing context. Typical
        # examples include weights and buffers.
        static_input_indices = tracing_context.fw_metadata.static_input_indices

    elif not is_forward:
        # For backward, identify saved tensors as static inputs, since saved
        # tensors are outputs of graph-wrapped forward runs. In PT2-generated
        # backward gm, saved tensors are always the leading args.
        fixed = count_tangents(gm)
        static_input_indices = list(range(fixed))

    return static_input_indices


def xpugraph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    is_forward: bool,
    static_input_indices: list[int] | None = None,
    tensor_input_indices: list[int] | None = None,
) -> torch.fx.GraphModule:
    """
    Apply xpugraph.

    This pass wraps the forward function with xpugraph during compilation and
    does not record xpugraph until runtime.

    - For the first run, it warms up operators and runtime state.
    - For the second run, it records xpugraph and replays xpugraph.
    - For following runs, it replays xpugraph.

    Args:
        gm: The graph module to wrap.
        example_inputs: Example inputs for warmup/recording.
        is_forward: Whether this is a forward graph or backward graph. Used to
            infer which inputs have stable tensor addresses when
            ``static_input_indices`` is not provided.
        static_input_indices: Explicit list of input indices with stable tensor
            addresses. When provided, ``is_forward`` is not used for inference.
        tensor_input_indices: Indices of graph inputs that are tensors, as
            opposed to opaque values like DeviceMesh. Used to compute which
            inputs need copying for xpugraph replay. When not provided, this is
            inferred from ``example_inputs``.
    """
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            f"xpugraph_pass requires a GraphModule but got {type(gm).__name__}. "
            f"Ensure xpugraph is not combined with passes that replace the "
            f"GraphModule, such as full_inductor_compilation."
        )

    if static_input_indices is None:
        static_input_indices = get_static_input_indices(gm, is_forward)

    gm.forward = XPUGraphWrapper(
        gm.forward,
        example_inputs,
        static_input_indices,
        tensor_input_indices=tensor_input_indices,
    )

    logger.info("Applied xpugraph pass.")
    return gm