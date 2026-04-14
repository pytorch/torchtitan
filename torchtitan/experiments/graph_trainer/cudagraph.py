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

import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._inductor.cudagraph_trees import _use_cuda_memory_pool_manager
from torch._library.opaque_object import is_opaque_value
from torch.utils._ordered_set import OrderedSet

logger = logging.getLogger(__name__)


class _CUDAGraphManager:
    """A manager to hold a shared graph pool, stream, and wrapper registry."""

    def __init__(self) -> None:
        self._initialized = False
        self._cudagraph_wrappers: list["CUDAGraphWrapper"] = []
        self._teardown_called = False

    def maybe_initialize(self) -> None:
        if self._initialized:
            return

        self._initialized = True

        # create a global cudagraph memory pool to allow memory reuse across cudagraphs.
        self.graph_pool = torch.cuda.graph_pool_handle()

        # create a global cuda stream for graph capture. we need to use a single stream
        # for all allocations to the memory pool, otherwise the allocations to separate
        # streams will not be used.
        self.stream = torch.cuda.Stream()

        # use a dummy graph to keep the global graph pool alive
        self._dummy_graph = torch.cuda.CUDAGraph()
        with (
            # suppress an empty cudagraph warning, since we intentionally create
            # an empty cudagraph here
            warnings.catch_warnings(record=True),
            torch.cuda.graph(
                self._dummy_graph,
                pool=self.graph_pool,
                stream=self.stream,
                capture_error_mode="thread_local",
            ),
        ):
            pass

    def register_wrapper(self, wrapper: "CUDAGraphWrapper") -> None:
        assert not self._teardown_called, "Cannot register new cudagraph after teardown"
        self._cudagraph_wrappers.append(wrapper)

    def teardown(self) -> None:
        """Destroy all cudagraphs and release the cudagraph memory pool.

        Note [explicit cudagraph teardown]
        cudagraph holds reference to nccl which prevents destroy process
        group. so we need to explicitly delete cudagraph which is held
        in _CUDAGraphManager and CUDAGraphWrapper. If cudagraph is not
        used, this is a no-op.
        """
        if not self._initialized:
            return
        if self._teardown_called:
            logger.warning("cudagraph manager teardown called twice")
            return

        for wrapper in self._cudagraph_wrappers:
            wrapper.teardown()
        self._cudagraph_wrappers.clear()

        self._dummy_graph = None
        self.stream = None
        self.graph_pool = None
        self._teardown_called = True


_cg_manager = _CUDAGraphManager()


def cudagraph_teardown() -> None:
    """Destroy all cudagraphs and release the cudagraph memory pool.
    See Note [explicit cudagraph teardown] for more details.
    """
    _cg_manager.teardown()


class CUDAGraphWrapper:
    """Wraps a callable with cudagraph. It warms up the callable, records cudagraph,
    and replays cudagraph during runtime. It also handles static input tensors, which
    are tensors whose tensor addresses do not change across runs.

    Args:
        runnable: The callable to wrap with CUDA graph. This can be a
            torch.fx.GraphModule when used in an FX graph pass, or any
            callable when used in PyTorch eager mode.
        example_inputs: A list of example inputs to the callable.
        static_input_indices: A tuple of indices identifying static input
            tensors. Static inputs are tensors whose memory addresses remain
            constant across invocations. Common examples include model weights,
            buffers, and outputs from previously wrapped CUDA graph functions.
        should_check_address: Whether to verify static input tensor addresses
            at runtime. This should only be enabled for debugging purposes.
    """

    def __init__(
        self,
        runnable: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: tuple[int] | None = None,
        should_check_address: bool = False,
    ):
        _cg_manager.maybe_initialize()
        _cg_manager.register_wrapper(self)

        self._runnable = runnable
        self._static_input_indices = OrderedSet(
            static_input_indices if static_input_indices is not None else []
        )
        self._input_indices_to_copy = [
            i
            for i, inp in enumerate(example_inputs)
            if isinstance(inp, torch.Tensor) and i not in self._static_input_indices
        ]
        self._cudagraph: torch.cuda.CUDAGraph | None = None
        self._has_warmup = False

        self._args = None
        self._output = None

        # (debug only) whether check static input tensor addresses during runtime
        self._should_check_address = should_check_address

        self._gm = runnable if isinstance(runnable, torch.fx.GraphModule) else None

    def print_readable(self, *args, **kwargs):
        """Delegate to the inner GraphModule's print_readable."""
        assert self._gm is not None, "print_readable requires a GraphModule runnable"
        return self._gm.print_readable(*args, **kwargs)

    def _copy_non_static_inputs(self, *args):
        for i in self._input_indices_to_copy:
            self._args[i].copy_(args[i])

    def _check_input_types(self, inputs) -> None:
        for i, inp in enumerate(inputs):
            if not (
                isinstance(inp, (torch.Tensor, int, float, torch._C.Generator))
                or is_opaque_value(inp)
            ):
                raise ValueError(
                    "args must be tensor, integer (for dynamic shapes), "
                    "float (for scalar constants), "
                    "Generator (for random number generator), "
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
        if not self._has_warmup:
            self._has_warmup = True
            device = torch.cuda.current_device()

            # warmup in cudagraph memory pool to avoid fragmentation
            # across eager memory pool and cudagraph memory pool.
            with _use_cuda_memory_pool_manager(
                device, _cg_manager.graph_pool, _cg_manager.stream
            ):
                out = self._runnable(*args)
            return out

        if self._cudagraph is None:
            self._check_input_types(args)
            self._args = args
            self._input_addresses = [
                x.data_ptr() if isinstance(x, torch.Tensor) else None for x in args
            ]

            self._cudagraph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(
                self._cudagraph,
                pool=_cg_manager.graph_pool,
                stream=_cg_manager.stream,
            ):
                # `output` is managed by pytorch's cudagraph pool
                self._output = self._runnable(*args)

        if self._should_check_address:
            self._check_static_inputs_address()

        self._copy_non_static_inputs(*args)
        self._cudagraph.replay()
        return self._output

    def teardown(self) -> None:
        """Destroy cudagraph and release references.
        See Note [explicit cudagraph teardown] for more details.
        """
        self._cudagraph = None
        self._args = None
        self._output = None


_FLEX_ATTENTION_OPS = {
    torch.ops.higher_order.flex_attention,
    torch.ops.higher_order.flex_attention_backward,
}


def is_cudagraph_compatible(gm: torch.fx.GraphModule) -> bool:
    """Check whether the graph can be safely captured by CUDA graph.

    Returns False (with a warning) when the graph contains patterns
    incompatible with CUDA graph capture:

    - **Unpinned CPU↔CUDA copies** (``aten.copy_``, ``aten._to_copy``):
      e.g. MoE load-balancing counters that copy tensors between CPU and
      CUDA.  CUDA graph capture requires pinned CPU memory for such copies.
    - **``aten._grouped_mm``**: the grouped matmul kernel used by MoE may
      perform internal CPU↔CUDA copies (e.g. workspace allocation) that
      are invisible in the FX graph metadata, breaking CUDA graph capture.
    - **flex_attention HOPs**: flex_attention higher-order ops require
      torch.compile (e.g. regional_inductor) to lower them into fused
      Triton kernels.  Without compilation they fall back to an unfused
      Math implementation that is incompatible with CUDA graph capture.
      The expected workflow is to apply regional_inductor first to compile
      flex_attention regions, then apply cudagraph.
    """
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        # Check for aten.copy_ / aten._to_copy between CPU and CUDA
        # without pin_memory (MoE load-balancing counters).
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
                        "Skipping cudagraph: graph contains unpinned CPU↔CUDA "
                        f"copy ({node.target})"
                    )
                    return False

        # Check for aten._grouped_mm unconditionally.
        # _grouped_mm may perform internal CPU↔CUDA copies (e.g. workspace
        # allocation) that are not visible from the FX graph metadata, so we
        # cannot rely on checking input device types alone.
        if node.target == torch.ops.aten._grouped_mm.default:
            logger.warning(
                "Skipping cudagraph: graph contains aten._grouped_mm "
                "which may perform internal CPU↔CUDA copies incompatible "
                "with CUDA graph capture"
            )
            return False

    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in _FLEX_ATTENTION_OPS:
            logger.warning(
                "Skipping cudagraph: graph contains flex_attention higher-order "
                "ops that require regional_inductor to compile before cudagraph "
                "can capture"
            )
            return False

    return True


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
