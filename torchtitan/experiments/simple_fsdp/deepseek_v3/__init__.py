# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.protocols.model_spec import ModelSpec

from .model import SimpleFSDPDeepSeekV3Model
from .parallelize import parallelize_deepseekv3


def _to_simple_fsdp_configs(base_configs):
    """Convert DeepSeekV3Model.Config instances to SimpleFSDPDeepSeekV3Model.Config."""
    return {
        k: SimpleFSDPDeepSeekV3Model.Config(
            **{f.name: getattr(v, f.name) for f in fields(v)}
        )
        for k, v in base_configs.items()
    }


_simple_fsdp_configs = _to_simple_fsdp_configs(deepseekv3_configs)


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="simple_fsdp/deepseek_v3",
        flavor=flavor,
        model=_simple_fsdp_configs[flavor],
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=None,
    )
