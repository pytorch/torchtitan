# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch._inductor.config
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

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


class ColwisePartitionOnly(ColwiseParallel):
    """ColwiseParallel that only partitions weights, without input/output hooks.

    Partitions weights identically to ColwiseParallel (Shard(0)), but does not
    register input/output hooks. The caller is responsible for handling DTensor
    conversion (e.g., using F.linear with weight.to_local()). This avoids
    DTensor dispatch wrapping x as DTensor(Replicate), whose backward would
    all-reduce d_x.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        else:
            raise NotImplementedError(
                "ColwisePartitionOnly currently only supports nn.Linear and nn.Embedding!"
            )
        module._partition_only = True
        return distribute_module(
            module,
            device_mesh,
            partition_fn,
        )


class RowwisePartitionOnly(RowwiseParallel):
    """RowwiseParallel that only partitions weights, without input/output hooks.

    Partitions weights identically to RowwiseParallel (Shard(1) for weight,
    Replicate for bias), but does not register input/output hooks. The caller
    is responsible for handling DTensor conversion. This avoids both the input
    hook's backward all-reduce on d_x and the output hook's Partial->Replicate
    all-reduce.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        else:
            raise NotImplementedError(
                "RowwisePartitionOnly currently only supports nn.Linear and nn.Embedding!"
            )
        module._partition_only = True
        return distribute_module(
            module,
            device_mesh,
            partition_fn,
        )
