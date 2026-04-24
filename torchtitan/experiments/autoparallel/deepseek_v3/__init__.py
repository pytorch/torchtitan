# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm

from torchtitan.models.deepseek_v3 import deepseekv3_configs, DeepSeekV3Model
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .parallelize_deepseekv3 import parallelize_deepseekv3


def model_registry(flavor: str) -> ModelSpec:
    config = deepseekv3_configs[flavor]()
    if "flex_attn" not in flavor:
        config.layers[0].attention = DeepSeekV3Model.Config().layers[0].attention
    return ModelSpec(
        name="autoparallel/deepseek_v3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
