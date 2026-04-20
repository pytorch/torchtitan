# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraphTrainerQwen3Model
from .parallelize import parallelize_qwen3


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    moe_comm_backend: str | None = None,
) -> ModelSpec:
    kwargs = dict(attn_backend=attn_backend)
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    base = qwen3_configs[flavor](**kwargs)
    config = GraphTrainerQwen3Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return ModelSpec(
        name="graph_trainer/qwen3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3,
        set_sharding_spec_fn=None,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
