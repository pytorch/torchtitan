# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
from typing import Any

import torch
import torch._inductor.config
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_module, DTensor, Replicate
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement
from torch.utils._pytree import tree_map

from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.tools.logging import logger


class NoParallel(ParallelStyle):
    """Replicate computation on the TP mesh without sharding.

    This style does nothing other than:
    (1) setting the module parameters as DTensors on the given mesh, and
    (2) inserting hooks at module boundary to convert torch.Tensor to DTensor and back.

    The reason we need this wrapping is to ensure all parameters are on the same 1D/2D mesh,
    which is assumed by (1) gradient norm clipping, and (2) optimizer fused implementation.

    Used for modules like the MoE router gate that need replicated computation on TP mesh.
    """

    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = False,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layout: Placement | None,
        desired_input_layout: Placement | None,
        mod: nn.Module,
        inputs: Any,
        device_mesh: DeviceMesh,
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            assert input_layout is not None
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            assert input_layout is not None
            assert desired_input_layout is not None
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(
        output_layout: Placement,
        use_local_output: bool,
        mod: nn.Module,
        outputs: Any,
        device_mesh: DeviceMesh,
    ) -> Any:
        # tree_map over outputs handles modules that return more than one value
        # (e.g. the router gate); non-DTensor outputs pass through unchanged.
        def _prepare(output: Any) -> Any:
            if not isinstance(output, DTensor):
                return output
            if output.placements != (output_layout,):
                output = output.redistribute(placements=(output_layout,), async_op=True)
            return output.to_local() if use_local_output else output

        return tree_map(_prepare, outputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn,  # pyrefly: ignore [bad-argument-type]
                self.input_layout,
                self.desired_input_layout,
            ),
            partial(
                self._prepare_output_fn,  # pyrefly: ignore [bad-argument-type]
                self.output_layout,
                self.use_local_output,
            ),
        )


def maybe_enable_async_tp(
    parallelism: ParallelismConfig, compile_config: CompileConfig, tp_mesh: DeviceMesh
):
    if not parallelism.enable_async_tensor_parallel:
        return

    if not (compile_config.enable and "model" in compile_config.components):
        raise RuntimeError(
            "Async TP requires 'model' in --compile.components and --compile.enable"
        )

    torch._inductor.config._micro_pipeline_tp = True

    logger.info("Async TP is enabled")
