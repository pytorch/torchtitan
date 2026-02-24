# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch._inductor.config
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.tools.logging import logger


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
