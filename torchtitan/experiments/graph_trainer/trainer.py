# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_teardown
from torchtitan.trainer import Trainer


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def close(self) -> None:
        super().close()

        # See Note [explicit cudagraph teardown] in cudagraph.py
        cudagraph_teardown()
