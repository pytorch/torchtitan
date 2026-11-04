# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer DeepSeek V4 model flavors."""

from dataclasses import fields

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4 import (
    build_model_config as build_deepseek_v4_model_config,
)

from ..common_utils import disable_inplace_wgrad_accum
from .model import GraphTrainerDeepSeekV4Model


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
) -> GraphTrainerDeepSeekV4Model.Config:
    base = build_deepseek_v4_model_config(
        flavor, parallelism=parallelism, seq_len=seq_len
    )
    config = GraphTrainerDeepSeekV4Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    # GraphTrainer traces the backward.
    disable_inplace_wgrad_accum(config)
    return config
