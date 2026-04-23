# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.experiments.autoparallel.graph_pipeline_parallel import (
    graph_pipeline_llm,
)
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.protocols.model_spec import ModelSpec

from .args import get_16b_sdpa_config, get_sample_config
from .model import DeepSeekV3Model  # noqa: F401 - triggers _owner wiring
from .parallelize_deepseekv3_graph_pp import parallelize_deepseekv3


def get_model_args():
    model_args = copy.deepcopy(deepseekv3_configs)

    # Preserve callbacks from the upstream 16B config for reuse
    update_from_config_16b = model_args["16B"].update_from_config
    get_nparams_and_flops_16b = model_args["16B"].get_nparams_and_flops

    # TODO: Align configs between AP and Titan
    for config in model_args.keys():
        # Just override the configs
        override = get_sample_config()
        override.update_from_config = model_args[config].update_from_config
        override.get_nparams_and_flops = model_args[config].get_nparams_and_flops
        model_args[config] = override

    # Add 16B_sdpa flavor (SDPA attention, Inductor-compatible)
    sdpa_config = get_16b_sdpa_config()
    sdpa_config.update_from_config = update_from_config_16b
    sdpa_config.get_nparams_and_flops = get_nparams_and_flops_16b
    model_args["16B_sdpa"] = sdpa_config

    return model_args


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
        pipelining_fn=graph_pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=None,
    )
