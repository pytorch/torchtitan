# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.gpt_oss import GptOssModel

from ..model import GraphTrainerModel


class GraphTrainerGptOssModel(GraphTrainerModel, GptOssModel):
    @dataclass(kw_only=True, slots=True)
    class Config(GptOssModel.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def parallelize(self, *, parallelism_context, **kwargs):
        # TODO: GPT-OSS CP needs its ContextParallelTransform attention
        # backends traced and validated under GraphTrainer.
        if parallelism_context.cp_enabled:
            raise ValueError(
                "Context parallelism is not supported for GraphTrainer GPT-OSS."
            )
        return super().parallelize(parallelism_context=parallelism_context, **kwargs)
