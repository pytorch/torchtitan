# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.experiments.graph_trainer.graph_pp.pipeline import graph_pipeline_llm
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from ..common_utils import build_decoder_config_for_backend
from .model import GraphTrainerQwen3Model
from .parallelize import parallelize_qwen3


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
) -> ModelSpec:
    kwargs = {}
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    get_config, max_context_len = qwen3_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    base = build_decoder_config_for_backend(
        get_config, attn_backend, seq_len=context_len, **kwargs
    )
    config = GraphTrainerQwen3Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return ModelSpec(
        name="graph_trainer/qwen3",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=graph_pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
