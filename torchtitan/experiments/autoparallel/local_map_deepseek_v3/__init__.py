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
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .args import get_sample_config
from .parallelize_deepseekv3 import parallelize_deepseekv3


def model_registry(flavor: str) -> ModelSpec:
    base = deepseekv3_configs[flavor]()
    # TODO: Align configs between AP and Titan
    override = get_sample_config()
    override.update_from_config = base.update_from_config
    override.get_nparams_and_flops = base.get_nparams_and_flops

    return ModelSpec(
        name="autoparallel/local_map_deepseek_v3",
        flavor=flavor,
        model=override,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
