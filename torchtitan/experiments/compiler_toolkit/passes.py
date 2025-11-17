# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compiler passes for the compiler toolkit.

This module provides various compiler passes that can be applied to graph modules
during compilation. Passes can be selected and configured via job config.
"""

from typing import Any, Callable, Optional, Sequence

import torch
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing
from torch.fx.passes.regional_inductor import regional_inductor
from torch.utils._ordered_set import OrderedSet


def autobucketing_reordering_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """
    Apply autobucketing and reordering optimization.

    This pass applies schedule_overlap_bucketing with collective_bucketing enabled
    to optimize communication patterns in distributed training.
    """
    schedule_overlap_bucketing(gm, collective_bucketing=True)
    gm.recompile()
    return gm


def regional_inductor_pass(
    gm: torch.fx.GraphModule, example_inputs
) -> torch.fx.GraphModule:
    """
    Apply regional inductor compilation based on user annotation.
    """
    return regional_inductor(gm, example_inputs)


_global_graph_pool = torch.cuda.graph_pool_handle()

# TODO: make output and args weakref to allow reuse.
# TODO: Check memory consumption


class CUDAGraphWrapper:
    def __init__(
        self,
        runnable: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: Optional[tuple[int]] = None,
    ):
        self.runnable = runnable
        self.graph_pool = _global_graph_pool
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

        # TODO: weak ref
        self.args = None
        self.output = None

    def copy_static_inputs(self, *args):
        for i in self.input_indices_to_copy:
            self.args[i].copy_(args[i])

    def __call__(self, *args):
        if not self.has_warmup:
            self.has_warmup = True
            return self.runnable(*args)

        if self.cudagraph is None:
            # TODO: weak ref?
            self.args = args
            input_addresses = [
                x.data_ptr() if isinstance(x, torch.Tensor) else None for x in args
            ]
            self.input_addresses = input_addresses

            self.cudagraph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(self.cudagraph, pool=self.graph_pool):
                # `output` is managed by pytorch's cudagraph pool
                # TODO: use weak ref for output to reuse memory
                self.output = self.runnable(*args)

        self.copy_static_inputs(*args)
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


# Registry mapping pass names to pass functions
AVAILABLE_PASSES = {
    "autobucketing_reordering": autobucketing_reordering_pass,
    "regional_inductor": regional_inductor_pass,
    "cudagraph": cudagraph_pass,
}
