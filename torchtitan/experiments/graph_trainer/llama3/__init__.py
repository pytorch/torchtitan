# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraphTrainerLlama3Model
from .parallelize import parallelize_llama


def _parallelize_fn(model, *, compile_config, loss_fn=None, **kwargs):
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

    if (
        isinstance(compile_config, GraphTrainerCompileConfig)
        and compile_config.autoparallel
    ):
        from .parallelize_autoparallel import parallelize_autoparallel_llama

        return parallelize_autoparallel_llama(
            model, compile_config=compile_config, loss_fn=loss_fn, **kwargs
        )
    return parallelize_llama(model, compile_config=compile_config, **kwargs)


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
) -> ModelSpec:
    base = llama3_configs[flavor](attn_backend=attn_backend)
    config = GraphTrainerLlama3Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return ModelSpec(
        name="graph_trainer/llama3",
        flavor=flavor,
        model=config,
        parallelize_fn=_parallelize_fn,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )
