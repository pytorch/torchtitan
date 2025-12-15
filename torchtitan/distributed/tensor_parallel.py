# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


def maybe_enable_async_tp(job_config: JobConfig, tp_mesh: DeviceMesh):
    if not job_config.parallelism.enable_async_tensor_parallel:
        return

    if not (job_config.compile.enable and "model" in job_config.compile.components):
        raise RuntimeError(
            "Async TP requires 'model' in --compile.components and --compile.enable"
        )

    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    torch._inductor.config._micro_pipeline_tp = True
    enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info("Async TP is enabled")
