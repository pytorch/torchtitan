# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraphTrainerDeepSeekV3Model
from .parallelize import parallelize_deepseekv3


def _parallelize_fn(model, *, compile_config, loss_fn=None, **kwargs):
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

    if (
        isinstance(compile_config, GraphTrainerCompileConfig)
        and compile_config.autoparallel
    ):
        from .parallelize_autoparallel import parallelize_autoparallel_deepseekv3

        return parallelize_autoparallel_deepseekv3(
            model, compile_config=compile_config, loss_fn=loss_fn, **kwargs
        )
    return parallelize_deepseekv3(model, compile_config=compile_config, **kwargs)


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
) -> ModelSpec:
    base = deepseekv3_configs[flavor](
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    config = GraphTrainerDeepSeekV3Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return ModelSpec(
        name="graph_trainer/deepseek_v3",
        flavor=flavor,
        model=config,
        parallelize_fn=_parallelize_fn,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
