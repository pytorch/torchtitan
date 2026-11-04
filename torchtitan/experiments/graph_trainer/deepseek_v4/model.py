# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.deepseek_v4 import DeepSeekV4Model

from ..model import GraphTrainerModel


class GraphTrainerDeepSeekV4Model(GraphTrainerModel, DeepSeekV4Model):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Model.Config):
        def __post_init__(self) -> None:
            DeepSeekV4Model.Config.__post_init__(self)
            # TODO: The traced loss receives the stacked labels tensor rather
            # than the per-depth labels tuple MTPCrossEntropyLoss needs.
            if self.n_mtp_layers > 0:
                raise ValueError(
                    "GraphTrainer DeepSeek V4 does not support MTP layers; got "
                    f"n_mtp_layers={self.n_mtp_layers}."
                )

    def __init__(self, config: Config):
        super().__init__(config)
