# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Sequence
from functools import partial

import torch
import torch._inductor.config
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_module, DTensor
from torch.distributed.tensor.parallel import ColwiseParallel
from torch.distributed.tensor.placement_types import Placement

from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.tools.logging import logger


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
