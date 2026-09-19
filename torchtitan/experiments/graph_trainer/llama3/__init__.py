# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.experiments.graph_trainer.graph_pp.pipeline import graph_pipeline_llm
from torchtitan.models.common.config_utils import TpGemmBackend
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from ..common_utils import build_decoder_config_for_backend
from .model import GraphTrainerLlama3Model
from .parallelize import parallelize_llama


def _parallelize_fn(model, *, compile_config, **kwargs):
    if compile_config.enable_autoparallel:
        from .parallelize_autoparallel import parallelize_autoparallel_llama

        return parallelize_autoparallel_llama(
            model, compile_config=compile_config, **kwargs
        )
    return parallelize_llama(model, compile_config=compile_config, **kwargs)


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    tp_gemm_backend: TpGemmBackend = "default",
) -> ModelSpec:
    get_config, max_context_len = llama3_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    base = build_decoder_config_for_backend(
        get_config, attn_backend, seq_len=context_len, tp_gemm_backend=tp_gemm_backend
    )
    config = GraphTrainerLlama3Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return ModelSpec(
        name="graph_trainer/llama3",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=_parallelize_fn,
        pipelining_fn=graph_pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )
