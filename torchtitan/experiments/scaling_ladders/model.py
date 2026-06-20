# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment-local Llama3 rung family for the scaling ladder.

Reuses ``Llama3Model`` and the public common builders; only the tiny per-layer
init dicts are reproduced locally to avoid coupling to llama3 module internals.
All rungs use untied embeddings and the Llama3 vocab, so ``lm_head`` is counted
in ``ladder_params = total_params - vocab_size * dim`` (OLMo-core's
``num_non_embedding_params``).
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

# Per-layer init dicts reproduced from llama3 (module-private there). Kept local
# so the ladder does not import llama3 internals; these are the standard llama3
# trunc-normal / ones / normal initializers plus depth-scaled output inits.
_LINEAR_INIT = {
    "weight": functools.partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": functools.partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": functools.partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": functools.partial(
            nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)
        ),
        "bias": nn.init.zeros_,
    }


@dataclass(frozen=True, kw_only=True)
class RungShape:
    """Architecture of one ladder rung. ``head_dim`` is always ``dim / n_heads``.

    ``nominal_params`` is the rung's size label (non-embedding params); ``None``
    skips the size audit (used by the debug rung).
    """

    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    hidden_dim: int
    nominal_params: int | None
    vocab_size: int = 128256


# Experiment-local rungs: hardware-friendly dims forming a smooth small-to-large
# ladder ending near Llama3-8B. Verified within +/-2% of the nominal label.
RUNGS: dict[str, RungShape] = {
    "60M": RungShape(
        dim=384,
        n_layers=5,
        n_heads=6,
        n_kv_heads=6,
        hidden_dim=1536,
        nominal_params=60_000_000,
    ),
    "100M": RungShape(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=8,
        hidden_dim=2048,
        nominal_params=100_000_000,
    ),
    "190M": RungShape(
        dim=768,
        n_layers=10,
        n_heads=12,
        n_kv_heads=12,
        hidden_dim=3072,
        nominal_params=190_000_000,
    ),
    "370M": RungShape(
        dim=1024,
        n_layers=14,
        n_heads=16,
        n_kv_heads=16,
        hidden_dim=4096,
        nominal_params=370_000_000,
    ),
    "760M": RungShape(
        dim=1536,
        n_layers=15,
        n_heads=24,
        n_kv_heads=24,
        hidden_dim=6144,
        nominal_params=760_000_000,
    ),
    "1B": RungShape(
        dim=2048,
        n_layers=11,
        n_heads=32,
        n_kv_heads=32,
        hidden_dim=8192,
        nominal_params=1_000_000_000,
    ),
    "3B": RungShape(
        dim=3072,
        n_layers=26,
        n_heads=24,
        n_kv_heads=8,
        hidden_dim=8192,
        nominal_params=3_000_000_000,
    ),
    "8B": RungShape(
        dim=4096,
        n_layers=34,
        n_heads=32,
        n_kv_heads=8,
        hidden_dim=14336,
        nominal_params=8_000_000_000,
    ),
    "debug": RungShape(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=16,
        hidden_dim=1024,
        nominal_params=None,
        vocab_size=2048,
    ),
}


def _build_layers(
    shape: RungShape, rope: ComplexRoPE.Config, attn_backend: str
) -> list[Llama3TransformerBlock.Config]:
    inner_attention = get_attention_config(attn_backend)
    n_kv_heads = None if shape.n_kv_heads == shape.n_heads else shape.n_kv_heads
    return [
        Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(
                normalized_shape=shape.dim, param_init=_NORM_INIT
            ),
            ffn_norm=RMSNorm.Config(normalized_shape=shape.dim, param_init=_NORM_INIT),
            attention=make_gqa_config(
                dim=shape.dim,
                n_heads=shape.n_heads,
                n_kv_heads=n_kv_heads,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(layer_id),
                inner_attention=inner_attention,
                rope=rope,
            ),
            feed_forward=make_ffn_config(
                dim=shape.dim,
                hidden_dim=shape.hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            ),
        )
        for layer_id in range(shape.n_layers)
    ]


def _build_rung_config(shape: RungShape, attn_backend: str) -> Llama3Model.Config:
    rope = ComplexRoPE.Config(
        dim=shape.dim // shape.n_heads,
        max_seq_len=131072,
        theta=500000,
        scaling="llama",
    )
    return Llama3Model.Config(
        dim=shape.dim,
        vocab_size=shape.vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=shape.vocab_size,
            embedding_dim=shape.dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=shape.dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=shape.dim,
            out_features=shape.vocab_size,
            param_init=_output_linear_init(shape.dim),
        ),
        layers=_build_layers(shape, rope, attn_backend),
    )


def model_registry(rung: str, attn_backend: str = "flex") -> ModelSpec:
    """Build the ``ModelSpec`` for a ladder rung, mirroring llama3's registry."""
    config = _build_rung_config(RUNGS[rung], attn_backend)
    return ModelSpec(
        name="scaling_ladders/llama3",
        flavor=rung,
        model=config,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


@functools.lru_cache(maxsize=None)
def count_total_params(rung: str) -> int:
    """Total params (including embeddings), built on meta.

    Backend-independent: attention backends add no parameters, so the count is
    built with a fixed backend regardless of the rung's configured one. Used for
    memory estimation (the optimizer state covers every parameter).
    """
    config = _build_rung_config(RUNGS[rung], "flex")
    with torch.device("meta"):
        model = config.build()
    return sum(p.numel() for p in model.parameters())


def count_ladder_params(rung: str) -> int:
    """Non-embedding params (untied, so lm_head is counted) = OLMo's N."""
    shape = RUNGS[rung]
    return count_total_params(rung) - shape.vocab_size * shape.dim
