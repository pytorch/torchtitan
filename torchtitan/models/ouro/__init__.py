# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config import LoopingConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common import (
    ComplexRoPE,
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from .model import OuroModel, OuroTransformerBlock
from .parallelize import parallelize_ouro

__all__ = [
    "OuroModel",
    "OuroTransformerBlock",
    "model_registry",
    "parallelize_ouro",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _build_ouro_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    rope: RoPE.Config,
    attn_backend: str,
) -> list[TransformerBlock.Config]:
    inner_attention = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            OuroTransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=1e-6,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=1e-6,
                    param_init=_NORM_INIT,
                ),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    rope=rope,
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _debugmodel(attn_backend: str) -> OuroModel.Config:
    dim = 256
    n_heads = 16
    vocab_size = 2048
    return OuroModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_ouro_layers(
            n_layers=6,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            rope=ComplexRoPE.Config(
                dim=dim // n_heads,
                max_seq_len=131072,
                theta=1_000_000,
            ),
            attn_backend=attn_backend,
        ),
    )


def _ouro_dense(
    *,
    n_layers: int,
    attn_backend: str,
) -> OuroModel.Config:
    dim = 2048
    n_heads = 16
    vocab_size = 49152
    return OuroModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_ouro_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=5632,
            rope=ComplexRoPE.Config(
                dim=dim // n_heads,
                max_seq_len=65536,
                theta=1_000_000,
            ),
            attn_backend=attn_backend,
        ),
    )


def _1_4b(attn_backend: str) -> OuroModel.Config:
    return _ouro_dense(n_layers=24, attn_backend=attn_backend)


def _2_6b(attn_backend: str) -> OuroModel.Config:
    return _ouro_dense(n_layers=48, attn_backend=attn_backend)


ouro_configs = {
    "debugmodel": _debugmodel,
    "1.4B": _1_4b,
    "2.6B": _2_6b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = ouro_configs[flavor](attn_backend=attn_backend)
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            c.build().convert(config)
    return ModelSpec(
        name="ouro",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_ouro,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def _base_trainer_config(
    *,
    flavor: str,
    hf_assets_path: str,
    seq_len: int,
    local_batch_size: int,
    steps: int,
) -> Trainer.Config:
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path=hf_assets_path,
        model_spec=model_registry(flavor),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2 if flavor == "debugmodel" else 2000,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=steps,
        ),
        looping=LoopingConfig(enable=True, steps=4, beta=0.05, exit_gate=True),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(enable_sequence_parallel=False),
        activation_checkpoint=SelectiveAC.Config(),
        validator=Validator.Config(enable=False),
    )


def ouro_debugmodel() -> Trainer.Config:
    return _base_trainer_config(
        flavor="debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
        seq_len=2048,
        local_batch_size=2,
        steps=10,
    )


def ouro_1_4b_r4() -> Trainer.Config:
    return _base_trainer_config(
        flavor="1.4B",
        hf_assets_path="./assets/hf/Ouro-1.4B",
        seq_len=4096,
        local_batch_size=1,
        steps=1000,
    )


def ouro_2_6b_r4() -> Trainer.Config:
    return _base_trainer_config(
        flavor="2.6B",
        hf_assets_path="./assets/hf/Ouro-2.6B",
        seq_len=4096,
        local_batch_size=1,
        steps=1000,
    )
