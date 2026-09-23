# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.muse_glimmer import MuseGlimmerModel

from ..model import GraphTrainerModel


class GraphTrainerMuseGlimmerModel(GraphTrainerModel, MuseGlimmerModel):
    @dataclass(kw_only=True, slots=True)
    class Config(MuseGlimmerModel.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def parallelize(self, *, parallel_dims, **kwargs):
        if parallel_dims.cp_enabled:
            raise ValueError(
                "Context parallelism is not supported for GraphTrainer MuseGlimmer."
            )
        return super().parallelize(parallel_dims=parallel_dims, **kwargs)
