# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.components.tokenizer import build_hf_tokenizer

from torchtitan.models.llama3 import pipeline_llama
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize_hf_transformers import parallelize_hf_transformers
from .model.hf_transformers_args import HFTransformerModelArgs, HFTransformerModel

from transformers.models.llama.modeling_llama import LlamaForCausalLM


__all__ = [
    "HFTransformerModelArgs",
    "HFTransformerModel",
    "hf_transformers_configs",
]


def hf_transformer_model_args_builder(**kwargs):
    # Capture the kwargs in the passed_args field
    args = HFTransformerModelArgs(**kwargs)
    args.passed_args = kwargs
    return args


flavors = {
    "debugmodel": hf_transformer_model_args_builder(
        # n_layers=2,
        # vocab_size=2000,
        max_seq_len=2048,
        #TODO(3outeille): n_kv_heads=n_heads may be handle somewhere else
        dim=256, n_layers=6, n_heads=16, vocab_size=2000, rope_theta=500000, n_kv_heads=16
    ),
    "medium": hf_transformer_model_args_builder(
        dim=1024,
        n_layers=12,
    ),
    "full": hf_transformer_model_args_builder(),
}

hf_train_spec = TrainSpec(
    name="hf_auto_model",
    model_cls=HFTransformerModel,
    model_args=flavors,
    parallelize_fn=parallelize_hf_transformers,
    pipelining_fn=pipeline_llama,
    build_optimizers_fn=build_optimizers,
    build_lr_schedulers_fn=build_lr_schedulers,
    build_dataloader_fn=build_hf_dataloader,
    build_tokenizer_fn=build_hf_tokenizer,
    build_loss_fn=build_cross_entropy_loss,
)

# Register multiple train_specs under the same name
register_train_spec(hf_train_spec)
register_train_spec(dataclasses.replace(hf_train_spec, name="meta-llama/Llama-3.2-3B"))
register_train_spec(dataclasses.replace(hf_train_spec, name="meta-llama/Llama-3.2-1B"))