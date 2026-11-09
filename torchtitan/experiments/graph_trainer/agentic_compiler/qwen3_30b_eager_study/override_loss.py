# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: per-chunk cross-entropy through the graph loop's loss kernels (G-90 / G-07).

    config.override.imports = ["override_loss.graph_kernel_loss"]
"""

from dataclasses import dataclass

import graph_kernels as gk

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size


class GraphKernelCrossEntropyLoss(CrossEntropyLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(CrossEntropyLoss.Config):
        pass

    def __call__(self, pred, labels, global_loss_token_counts=None, **kwargs):
        if (
            self.z_loss_weight is not None
            or global_loss_token_counts is None
            or spmd_mesh_size("tp") > 1
        ):
            raise ValueError("graph-kernel loss: unsupported configuration")
        return gk.CrossEntropyChunk.apply(pred, labels, global_loss_token_counts), {}


@override(
    target=CrossEntropyLoss.Config,
    description="Cross-entropy chunk through the graph loop's verified kernels",
)
def graph_kernel_loss(cfg: CrossEntropyLoss.Config):
    return derive(cfg, GraphKernelCrossEntropyLoss.Config)
