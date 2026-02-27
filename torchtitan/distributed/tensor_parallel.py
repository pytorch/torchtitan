# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Sequence
from functools import partial
from typing import Any

import torch
import torch._inductor.config
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_module, DTensor, Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle
from torch.distributed.tensor.placement_types import Placement

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
        local_output_grad_placements: Sequence[Placement] | None = None,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        # If None, output stays as DTensor.
        # If provided, output is cast to local tensor via
        # to_local(grad_placements=local_output_grad_placements).
        self.local_output_grad_placements = local_output_grad_placements

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
        local_output_grad_placements: Sequence[Placement] | None,
        mod: nn.Module,
        outputs: DTensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor | DTensor:
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        if local_output_grad_placements is not None:
            return outputs.to_local(grad_placements=local_output_grad_placements)
        else:
            return outputs

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
                self.local_output_grad_placements,
            ),
        )


class ColwiseParallelWithGradPlacement(ColwiseParallel):
    """ColwiseParallel with explicit control over backward gradient placement.

    By default, ``ColwiseParallel`` with ``input_layouts=Replicate()`` wraps
    the input via ``from_local(Replicate)``, whose backward all-reduces d_x
    back to Replicate.  This subclass overrides ``_prepare_input_fn`` to pass
    ``local_input_grad_placements`` to ``DTensor.from_local``, giving users
    explicit control over the gradient placement during backward.  When not
    specified, defaults to ``None`` and the gradient placement follows the
    default guarantees of ``DTensor.from_local``.
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
        local_input_grad_placements: Sequence[Placement] | None = None,
    ):
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )
        self.local_input_grad_placements = local_input_grad_placements

    @staticmethod
    def _prepare_input_fn(  # pyrefly: ignore [bad-param-name-override]
        input_layouts,
        desired_input_layouts,
        local_input_grad_placements,
        mod,
        inputs,
        device_mesh,
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            assert local_input_grad_placements is not None, (
                "local_input_grad_placements must be specified when input is a "
                "plain tensor. Please think about what you want the from_local(Replicate) backward behavior like."
            )
            input_tensor = DTensor.from_local(
                input_tensor,
                device_mesh,
                input_layouts,
                run_check=False,
                grad_placements=local_input_grad_placements,
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        else:
            raise NotImplementedError(
                "ColwiseParallelWithGradPlacement currently only supports nn.Linear and nn.Embedding!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn,  # pyrefly: ignore [bad-argument-type]
                self.input_layouts,
                self.desired_input_layouts,
                self.local_input_grad_placements,
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
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
