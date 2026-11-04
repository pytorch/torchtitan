# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.kimi_k3 import KimiK3Model

from ..model import GraphTrainerModel


class GraphTrainerKimiK3Model(GraphTrainerModel, KimiK3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(KimiK3Model.Config):
        def __post_init__(self) -> None:
            KimiK3Model.Config.__post_init__(self)
            # TODO: The vision encoder reads image grids on the host and builds
            # per-batch shapes, which one static-shape trace cannot capture.
            if self.vision_encoder is not None:
                raise ValueError(
                    "GraphTrainer Kimi K3 supports text-only flavors; got a "
                    "vision encoder."
                )

    def __init__(self, config: Config):
        super().__init__(config)

    def parallelize(self, *, parallelism_context, **kwargs):
        # TODO: Kimi K3 CP needs its ContextParallelTransform attention
        # backends traced and validated under GraphTrainer.
        if parallelism_context.cp_enabled:
            raise ValueError(
                "Context parallelism is not supported for GraphTrainer Kimi K3."
            )
        return super().parallelize(parallelism_context=parallelism_context, **kwargs)

    def pipeline(self, **kwargs):
        # TODO: GraphPP does not route Kimi K3's attention-residual blocks
        # between stages (see pipeline_kimi_k3).
        raise ValueError(
            "Pipeline parallelism is not supported for GraphTrainer Kimi K3."
        )
