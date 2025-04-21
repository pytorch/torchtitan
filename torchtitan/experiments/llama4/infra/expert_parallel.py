# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement


# implementation of Tensor Parallel for the GroupedExperts in MoE
class TensorParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Tuple[Optional[Placement]]] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = input_layouts or (Replicate(), Replicate())
        self.output_layout = output_layout or Replicate()
        self.desired_input_layouts = (Replicate(), Replicate())
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        prepared_inputs = []
        # annotate module input placements/sharding with input_layouts
        for inp, input_layout, desired_input_layout in zip(
            inputs, input_layouts, desired_input_layouts
        ):
            if isinstance(inp, torch.Tensor):
                if not isinstance(inp, DTensor):
                    inp = DTensor.from_local(
                        inp, device_mesh, (input_layout,), run_check=False
                    )
                if input_layout != desired_input_layout:
                    inp = inp.redistribute(
                        placements=(desired_input_layout,), async_op=True
                    )
            prepared_inputs.append(inp)
        return tuple(prepared_inputs)

    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(2)]))
        )  # Column-wise sharding
        module.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(1)])),
        )  # Row-wise sharding
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(2)])),
        )  # Column-wise sharding

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


# NOTE: This is to achieve replicate computation on the gate module in the MoE router.
# It does nothing other than (1) setting the module parameters as DTensors on the given mesh
# and (2) inserting hooks to module boundary to change torch.Tensor to DTensor and back.
# TODO: The reason we need this wrapping is to ensure all parameters are on the same 1D/2D mesh,
# which is assumed by (1) gradient norm clipping, and (2) optimizer fused implementation.
class NoParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn, self.input_layout, self.desired_input_layout
            ),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )
