# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.kimi_k2_7 import KimiK25Model
from torchtitan.models.kimi_k2_7.qk_clip import QKClipFlexInnerAttention

from ..model import GraphTrainerModel


class GraphTrainerKimiK25Model(GraphTrainerModel, KimiK25Model):
    @dataclass(kw_only=True, slots=True)
    class Config(KimiK25Model.Config):
        def __post_init__(self) -> None:
            KimiK25Model.Config.__post_init__(self)
            # TODO: The vision encoder reads image grids on the host and builds
            # per-batch shapes, which one static-shape trace cannot capture.
            if self.vision_encoder is not None:
                raise ValueError(
                    "GraphTrainer Kimi K2.7 supports text-only flavors; got a "
                    "vision encoder."
                )
            # TODO: QK clipping appends each step's attention logits to a Python
            # list that the optimizer hook reads; a traced step does not replay
            # the append. It needs the logits as a graph output or buffer.
            if next(self.traverse(QKClipFlexInnerAttention.Config), None) is not None:
                raise ValueError(
                    "GraphTrainer Kimi K2.7 does not support QK clipping; build "
                    "the model with graph_trainer.kimi_k2_7.build_model_config."
                )

    def __init__(self, config: Config):
        super().__init__(config)

    def parallelize(self, *, parallelism_context, **kwargs):
        # GraphTrainerModel.parallelize takes precedence over KimiK25Model's,
        # so repeat its CP check.
        if parallelism_context.cp_enabled:
            raise ValueError(
                "Context parallelism is not supported for GraphTrainer Kimi K2.7."
            )
        return super().parallelize(parallelism_context=parallelism_context, **kwargs)
