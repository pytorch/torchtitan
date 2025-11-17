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


from typing import Callable, Optional

_global_graph_pool = torch.cuda.graph_pool_handle()

# TODO: make output and args weakref to allow reuse.


class CUDAGraphWrapper:
    def __init__(
        self,
        runnable: Callable,
        static_input_indices: Optional[tuple[int]] = None,
    ):
        self.runnable = runnable
        self.graph_pool = _global_graph_pool
        self.static_input_indices = OrderedSet(
            static_input_indices if static_input_indices is not None else []
        )

        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None

        # TODO: weak ref
        self.args = None
        self.kwargs = None
        self.output = None

        self.has_warmup = False

    def copy_static_inputs(self, *args):
        for i in range(len(self.args)):
            if i not in self.static_input_indices and isinstance(self.args[i], torch.Tensor):
                self.args[i].copy_(args[i])

    def __call__(self, *args, **kwargs):
        if not self.has_warmup:
            self.has_warmup = True
            return self.runnable(*args, **kwargs)

        if self.cudagraph is None:
            # TODO: weak ref?
            self.args = args
            self.kwargs = kwargs
            input_addresses = [
                x.data_ptr() if isinstance(x, torch.Tensor) else None for x in args 
            ]
            self.input_addresses = input_addresses

            self.cudagraph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(self.cudagraph, pool=self.graph_pool):
                # `output` is managed by pytorch's cudagraph pool
                # TODO: use weak ref for output to reuse memory
                self.output = self.runnable(*args, **kwargs)

        # if True:
        #     # check if the input addresses are the same
        #     new_input_addresses = [
        #         x.data_ptr() for x in args if isinstance(x, torch.Tensor)
        #     ]
        #     assert new_input_addresses == self.input_addresses, (
        #         f"Input addresses for cudagraphs are different "
        #         f"during replay. Expected {self.input_addresses}, "
        #         f"got {new_input_addresses}"
        #     )


        self.copy_static_inputs(*args)
        self.cudagraph.replay()
        return self.output


def cudagraph_pass(
    gm: torch.fx.GraphModule, example_inputs
) -> torch.fx.GraphModule:
    """
    Apply cudagraph.

    This pass wraps the forward function with cudagraph during compilation and does
    not record cudagraph until runtime.
    - For the first run, it will warm up operators such as nccl.
    - For the second run, it will record cudagraph and replay cudagraph.
    - For the following runs, it will replay cudagraph.
    """
    gm.forward = CUDAGraphWrapper(gm.forward)
    return gm


# Registry mapping pass names to pass functions
AVAILABLE_PASSES = {
    "autobucketing_reordering": autobucketing_reordering_pass,
    "regional_inductor": regional_inductor_pass,
    "cudagraph": cudagraph_pass,
}


# TODO: cleanup graph before nccl destroy group
