# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraphTrainerLlama3Model
from .parallelize import parallelize_llama


def _to_graph_trainer_configs(base_configs):
    """Convert Llama3Model.Config instances to GraphTrainerLlama3Model.Config."""
    return {
        k: GraphTrainerLlama3Model.Config(
            **{f.name: getattr(v, f.name) for f in fields(v)}
        )
        for k, v in base_configs.items()
    }


_graph_trainer_configs = _to_graph_trainer_configs(llama3_configs)


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="graph_trainer/llama3",
        flavor=flavor,
        model=_graph_trainer_configs[flavor],
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )
