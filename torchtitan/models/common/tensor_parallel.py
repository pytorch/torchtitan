# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model components that own their tensor-parallel collectives."""

from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.feed_forward import FeedForward


class TensorParallelFeedForward(FeedForward):
    """Feed-forward whose projection regions own the standard TP collectives.

    With sequence parallelism, ``w13`` all-gathers the input token dimension
    and ``w2`` reduce-scatters its partial output. Without sequence parallelism,
    the corresponding redistributions are ``I -> R`` and ``P -> I``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        enable_sequence_parallel: bool = True

    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_sequence_parallel = config.enable_sequence_parallel

    def _boundary_type(self) -> spmd.PerMeshAxisSpmdType:
        return spmd.S(0) if self.enable_sequence_parallel else spmd.I

    def _project_w13(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather the input and run w13 inside one remat region."""
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            x = spmd.redistribute(
                x,
                tp_group,
                src=self._boundary_type(),
                dst=spmd.R,
                backward_options={"op_dtype": x.dtype},
            )
        return super()._project_w13(x)

    def _project_w2(self, x: torch.Tensor) -> torch.Tensor:
        """Run w2 and reduce its output inside one remat region."""
        out_TD = super()._project_w2(x)
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is None:
            return out_TD
        return spmd.redistribute(
            out_TD,
            tp_group,
            src=spmd.P,
            dst=self._boundary_type(),
            backward_options={"op_dtype": out_TD.dtype},
        )


__all__ = ["TensorParallelFeedForward"]
