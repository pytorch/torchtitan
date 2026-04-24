# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.graph_trainer.compile import graph_pp_pipeline_llm
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.protocols.model_spec import ModelSpec

from .args import get_sample_config
from .model import DeepSeekV3Model  # noqa: F401 - triggers _owner wiring
from .parallelize import parallelize_deepseekv3


def _pipelining_fn(model, *, compile_config, **kwargs):
    """Dispatch to graph PP when using GraphTrainerCompileConfig, else standard PP."""
    if isinstance(compile_config, GraphTrainerCompileConfig):
        return graph_pp_pipeline_llm(model, compile_config=compile_config, **kwargs)
    return pipeline_llm(model, compile_config=compile_config, **kwargs)


def model_registry(flavor: str) -> ModelSpec:
    base = deepseekv3_configs[flavor]()
    # TODO: Align configs between AP and Titan
    override = get_sample_config()
    override.update_from_config = base.update_from_config
    override.get_nparams_and_flops = base.get_nparams_and_flops

    return ModelSpec(
        name="graph_trainer/autoparallel_deepseek_v3",
        flavor=flavor,
        model=override,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=_pipelining_fn,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=None,
    )
