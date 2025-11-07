# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.dataloader import build_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from ..llama3 import (
    Llama3StateDictAdapter,
    parallelize_llama,
    Transformer,
    TransformerModelArgs,
)

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "qwen2_configs",
]

# Adding different variants of the model

qwen2_configs = {
    "7B": TransformerModelArgs(
        vocab_size=152064,
        max_seq_len=131072,
        head_dim=128,
        dim=3584,
        hidden_dim=18944,
        n_layers=28,
        norm_eps=1e-6,
        n_heads=28,
        n_kv_heads=4,
        use_qkv_bias=True,
        rope_theta=1000000,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=qwen2_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
