# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_16b
from torchtitan.trainer import Trainer

from . import model_registry


def flex_shard_deepseek_v3_16b_dp8() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_spec = model_registry("16B", attn_backend="flex")
    return config
