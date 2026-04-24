# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.graph_trainer.compile import graph_pp_pipeline_llm
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraphTrainerDeepSeekV3Model
from .parallelize import parallelize_deepseekv3


def _pipelining_fn(model, *, compile_config, **kwargs):
    """Dispatch to graph PP when using GraphTrainerCompileConfig, else standard PP."""
    if isinstance(compile_config, GraphTrainerCompileConfig):
        return graph_pp_pipeline_llm(model, compile_config=compile_config, **kwargs)
    return pipeline_llm(model, compile_config=compile_config, **kwargs)


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
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=_pipelining_fn,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
