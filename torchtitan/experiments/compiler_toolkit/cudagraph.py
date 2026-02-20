# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CUDAGraph pass for the compiler toolkit.

This module provides a cudagraph pass that can be applied to graph modules
during compilation.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
from torch._inductor.cudagraph_trees import _use_cuda_memory_pool_manager
from torch.utils._ordered_set import OrderedSet


def init_global_graph_pool() -> tuple[
    torch.cuda.CUDAGraph, torch.cuda._POOL_HANDLE, torch.cuda.Stream
]:
    dummy_graph = torch.cuda.CUDAGraph()

    # create a global cudagraph memory pool to allow memory reuse across cudagraphs.
    graph_pool = torch.cuda.graph_pool_handle()

    # create a global cuda stream for graph capture. we need to use a single stream
    # for all allocations to the memory pool, otherwise the allocations to separate streams
    # will not be used.
    graph_capture_stream = torch.cuda.Stream()

    # use a dummy graph to keep the global graph pool alive
    with (
        # suppress an empty cudagraph warning, since we intentionally create
        # an empty cudagraph here
        warnings.catch_warnings(record=True),
        torch.cuda.graph(
            dummy_graph,
            pool=graph_pool,
            stream=graph_capture_stream,
            capture_error_mode="thread_local",
        ),
    ):
        pass

    return dummy_graph, graph_pool, graph_capture_stream


(
    _global_dummy_graph,
    _global_graph_pool,
    _global_graph_capture_stream,
) = init_global_graph_pool()


class CUDAGraphWrapper:
    def __init__(
        self,
        runnable: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: Optional[tuple[int]] = None,
        should_check_address: bool = False,
    ):
        self.runnable = runnable
        self.graph_pool = _global_graph_pool
        self.stream = _global_graph_capture_stream
        self.static_input_indices = OrderedSet(
            static_input_indices if static_input_indices is not None else []
        )
        self.input_indices_to_copy = [
            i
            for i, inp in enumerate(example_inputs)
            if isinstance(inp, torch.Tensor) and i not in self.static_input_indices
        ]
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self.has_warmup = False

        self.args = None
        self.output = None

        # (debug only) whether check static input tensor addresses during runtime
        self.should_check_address = should_check_address

    def copy_non_static_inputs(self, *args):
        for i in self.input_indices_to_copy:
            self.args[i].copy_(args[i])

    def check_input_types(self, inputs) -> None:
        for inp in inputs:
            assert isinstance(inp, (torch.Tensor, int, torch._C.Generator)), (
                "args must be tensor, integer (for dynamic shapes), "
                "or Generator (for random number generator), "
                f"but found {type(inp)}"
            )

    def check_static_inputs_address(self) -> None:
        for i in self.static_input_indices:
            actual = self.args[i].data_ptr()
            expected = self.input_addresses[i]
            assert expected == actual, (
                "Expected the same static tensor address but found "
                f"{expected} != {actual}"
            )

    def __call__(self, *args):
        if not self.has_warmup:
            self.has_warmup = True
            device = torch.cuda.current_device()

            # warmup in cudagraph memory pool to avoid fragmentation
            # across eager memory pool and cudagraph memory pool.
            with _use_cuda_memory_pool_manager(device, self.graph_pool, self.stream):
                out = self.runnable(*args)
            return out

        if self.cudagraph is None:
            self.check_input_types(args)
            self.args = args
            self.input_addresses = [
                x.data_ptr() if isinstance(x, torch.Tensor) else None for x in args
            ]

            self.cudagraph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(
                self.cudagraph, pool=self.graph_pool, stream=self.stream
            ):
                # `output` is managed by pytorch's cudagraph pool
                self.output = self.runnable(*args)

        if self.should_check_address:
            self.check_static_inputs_address()

        self.copy_non_static_inputs(*args)
        self.cudagraph.replay()
        return self.output


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
