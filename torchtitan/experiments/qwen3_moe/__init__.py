# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.components.tokenizer import build_hf_tokenizer

# ToDO - this is not suitable for qwen but using for now...
from torchtitan.models.llama3 import pipeline_llama
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize_qwen import parallelize_qwen

from torchtitan.experiments.qwen3_moe.model.model import QwenForCausalLM

from torchtitan.experiments.qwen3_moe.model.args import  Qwen3MoE


__all__ = [
    "Qwen3MoE",
    "QwenForCausalLM",
    "qwen_configs",
]


qwen_configs = {
    "debugmodel": Qwen3MoE(
        dim=2048,
        n_layers=48,
        n_heads=32,
        n_kv_heads=4,
        rope_theta=10000000,
        num_experts=128,
        top_k=8,
        vocab_size=151936,
    ),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": Qwen3MoE(
        dim=2048,     # hidden_size from HF config
        n_layers=48,  # num_hidden_layers from HF config
        n_heads=32,   # num_attention_heads from HF config
        n_kv_heads=4, # num_key_value_heads from HF config
        rope_theta=10000000,
        num_experts=128,  # num_experts from HF config
        top_k=8,         # num_experts_per_tok from HF config
        vocab_size=151936,  # vocab_size from HF config
    ),
}


register_train_spec(
    TrainSpec(
        name="qwen3_moe",
        model_cls=QwenForCausalLM,
        model_args=qwen_configs,
        parallelize_fn=parallelize_qwen,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
