# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import copy

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .args import get_sample_config
from .parallelize_deepseekv3 import parallelize_deepseekv3


def get_model_args():
    model_args = copy.deepcopy(deepseekv3_configs)
    # TODO: Align configs between AP and Titan
    for config in model_args.keys():
        # Just override the configs
        override = get_sample_config()
        override.update_from_config = model_args[config].update_from_config
        override.get_nparams_and_flops = model_args[config].get_nparams_and_flops
        model_args[config] = override

    return model_args


def model_registry(flavor: str) -> ModelSpec:
    model_args = get_model_args()

    return ModelSpec(
        name="autoparallel/local_map_deepseek_v3",
        flavor=flavor,
        model=model_args[flavor],
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
