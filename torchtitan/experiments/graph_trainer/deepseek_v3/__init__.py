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
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraphTrainerDeepSeekV3Model
from .parallelize import parallelize_deepseekv3


def _to_graph_trainer_configs(base_configs):
    """Convert DeepSeekV3Model.Config instances to GraphTrainerDeepSeekV3Model.Config."""
    return {
        k: GraphTrainerDeepSeekV3Model.Config(
            **{f.name: getattr(v, f.name) for f in fields(v)}
        )
        for k, v in base_configs.items()
    }


_graph_trainer_configs = _to_graph_trainer_configs(deepseekv3_configs)


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="graph_trainer/deepseek_v3",
        flavor=flavor,
        model=_graph_trainer_configs[flavor],
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
